import boto3
import json
import streamlit as st
import pandas as pd
import traceback
import decimal
import time
from datetime import datetime
import yaml
import os
from typing import Dict, List, Any, Optional
from botocore.config import Config
from feedback_component import FeedbackComponent

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="Container Migration Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to change the slider color from red to a more neutral blue
st.markdown("""
<style>
    /* Change slider color from red to blue */
    .stSlider [data-baseweb="slider"] {
        background-color: rgba(151, 166, 195, 0.25);
    }
    .stSlider [data-baseweb="thumb"] {
        background-color: #4e8df5 !important;
    }
    /* Change the active track color */
    .stSlider [data-baseweb="track"] div div {
        background-color: #4e8df5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Decimal objects."""
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            # Convert decimal to float for JSON serialization
            return float(obj)
        # Let the base class default method handle other types
        return super(DecimalEncoder, self).default(obj)

# Helper function to safely format timestamps
def format_timestamp(timestamp_value):
    """
    Safely format a timestamp to YYYY-MM-DD HH:MM format.
    
    Args:
        timestamp_value: The timestamp value which could be a string, Timestamp object, or other
        
    Returns:
        str: Formatted timestamp string
    """
    if timestamp_value == 'N/A':
        return timestamp_value
        
    # Convert to string to handle various timestamp types
    timestamp_str = str(timestamp_value)
    
    try:
        # Direct string manipulation to ensure format
        if ' ' in timestamp_str:
            parts = timestamp_str.split(' ')
            if len(parts) >= 2:
                date_part = parts[0]
                time_parts = parts[1].split(':')
                if len(time_parts) >= 2:
                    time_part = f"{time_parts[0]}:{time_parts[1]}"
                    return f"{date_part} {time_part}"
    except Exception as e:
        logging.error(f"Error formatting timestamp {timestamp_str}: {e}")
    
    # Return original if formatting fails
    return timestamp_str

class DashboardApp:
    """Dashboard application for container migration from EC2 to EKS."""
    
    def __init__(self):
        """Initialize the dashboard with AWS service clients."""
        # Get region from environment variable or default to us-east-1
        self.region = os.environ.get('AWS_REGION', 'us-east-1')

        # Initialize AWS service clients
        self.s3 = boto3.client('s3', region_name=self.region)
        self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
        self.lambda_client = boto3.client('lambda', region_name=self.region)
        
        # Initialize DynamoDB table
        self.migration_table = self.dynamodb.Table(
            os.environ.get('MIGRATION_TABLE_NAME', 'eks-migration-zetaglobal')
        )
        
        # S3 configuration
        self.artifact_bucket = os.environ.get('ARTIFACT_BUCKET', 'eks-migration-zetaglobal')
        
        # Lambda function name
        self.supervisor_lambda = os.environ.get('SUPERVISOR_LAMBDA', 'eks-migration-zetaglobal-supervisor')
        
        config = Config(read_timeout=1000)
        self.bedrock = boto3.client('bedrock-runtime', region_name=self.region, config = config)
        self.bedrock_agent = boto3.client('bedrock-agent-runtime', region_name=self.region, config = config)
        self.model_id = os.environ.get('BEDROCK_MODEL_ID', 'us.anthropic.claude-3-7-sonnet-20250219-v1:0')
        self.kb_id = os.environ.get('BEDROCK_KB_ID', 'JZJOWPRCEH')
    
    def load_migration_data(self) -> pd.DataFrame:
        """
        Load migration data from DynamoDB.
        
        Returns:
            pd.DataFrame: DataFrame containing migration data
        """
        # Check if we have cached data and should use it
        if 'migration_data_cache' in st.session_state and not st.session_state.get('force_refresh', False):
            return st.session_state.migration_data_cache
            
        try:
            # Scan the migration table
            response = self.migration_table.scan()
            items = response.get('Items', [])
            
            # Handle pagination if needed
            while 'LastEvaluatedKey' in response:
                response = self.migration_table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))
            
            # Convert to DataFrame
            if items:
                # Convert Decimal to float for all items
                for item in items:
                    for key, value in item.items():
                        if isinstance(value, decimal.Decimal):
                            item[key] = float(value)
                
                df = pd.DataFrame(items)
                
                # Convert created_at and updated_at to datetime
                for col in ['created_at', 'updated_at']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Ensure status and current_stage are strings
                if 'status' in df.columns:
                    df['status'] = df['status'].astype(str)
                if 'current_stage' in df.columns:
                    df['current_stage'] = df['current_stage'].astype(str)
                
                # Cache the data for future use
                st.session_state.migration_data_cache = df
                
                # Reset force_refresh flag
                if 'force_refresh' in st.session_state:
                    st.session_state.force_refresh = False
                
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading migration data: {str(e)}")
            st.code(traceback.format_exc())
            return pd.DataFrame()
    
    def load_artifacts(self, migration_id: str) -> Dict:
        """
        Load artifacts for a migration from S3.
        
        Args:
            migration_id: Migration ID
            
        Returns:
            Dict: Dictionary of artifacts by agent type
        """
        # Check if we have a cached version of the artifacts
        if 'artifacts_cache' not in st.session_state:
            st.session_state.artifacts_cache = {}
            
        # Return cached artifacts if available and not forcing refresh
        if migration_id in st.session_state.artifacts_cache and not st.session_state.get('force_refresh', False):
            return st.session_state.artifacts_cache[migration_id]
            
        try:
            # List objects with the migration_id prefix
            response = self.s3.list_objects_v2(
                Bucket=self.artifact_bucket,
                Prefix=f"artifacts/{migration_id}/"
            )
            
            artifacts = {}
            if 'Contents' in response:
                for item in response['Contents']:
                    key = item['Key']
                    parts = key.split('/')
                    
                    # Handle both old and new structures
                    if len(parts) >= 4:
                        agent_type = parts[2]
                        
                        # Check if this is multi-app structure (base/charts/app_name/...)
                        if len(parts) >= 6 and parts[3] == 'base' and parts[4] == 'charts':
                            # Multi-app structure: artifacts/migration_id/container_config/base/charts/app_name/...
                            app_name = parts[5]
                            if len(parts) >= 7:
                                artifact_name = f"{app_name}/{'/'.join(parts[6:])}"
                            else:
                                artifact_name = app_name
                        elif len(parts) >= 5 and parts[3] == 'base':
                            # Umbrella chart files: artifacts/migration_id/container_config/base/Chart.yaml
                            artifact_name = f"base/{'/'.join(parts[4:])}"
                        else:
                            # Single app or old structure: artifacts/migration_id/agent_type/filename
                            artifact_name = '/'.join(parts[3:])
                        
                        if agent_type not in artifacts:
                            artifacts[agent_type] = []
                        
                        artifacts[agent_type].append({
                            'name': artifact_name,
                            'key': key,
                            'last_modified': item['LastModified'],
                            'size': item['Size']
                        })
            
            # Handle pagination if needed
            while response.get('IsTruncated', False):
                response = self.s3.list_objects_v2(
                    Bucket=self.artifact_bucket,
                    Prefix=f"artifacts/{migration_id}/",
                    ContinuationToken=response['NextContinuationToken']
                )
                
                if 'Contents' in response:
                    for item in response['Contents']:
                        key = item['Key']
                        parts = key.split('/')
                        
                        if len(parts) >= 4:
                            agent_type = parts[2]
                            
                            # Check if this is multi-app structure
                            if len(parts) >= 6 and parts[3] == 'base' and parts[4] == 'charts':
                                app_name = parts[5]
                                if len(parts) >= 7:
                                    artifact_name = f"{app_name}/{'/'.join(parts[6:])}"
                                else:
                                    artifact_name = app_name
                            elif len(parts) >= 5 and parts[3] == 'base':
                                artifact_name = f"base/{'/'.join(parts[4:])}"
                            else:
                                artifact_name = '/'.join(parts[3:])
                            
                            if agent_type not in artifacts:
                                artifacts[agent_type] = []
                            
                            artifacts[agent_type].append({
                                'name': artifact_name,
                                'key': key,
                                'last_modified': item['LastModified'],
                                'size': item['Size']
                            })
            
            # Cache the artifacts for future use
            st.session_state.artifacts_cache[migration_id] = artifacts
            
            return artifacts
        except Exception as e:
            st.error(f"Error loading artifacts: {str(e)}")
            st.code(traceback.format_exc())
            return {}
    
    def get_artifact_content(self, key: str) -> str:
        """
        Get the content of an artifact from S3.
        
        Args:
            key: S3 key of the artifact
            
        Returns:
            str: Content of the artifact
        """
        try:
            # Check if bucket exists first
            try:
                self.s3.head_bucket(Bucket=self.artifact_bucket)
            except Exception as bucket_error:
                return f"Error: S3 bucket '{self.artifact_bucket}' does not exist or you don't have access to it: {str(bucket_error)}"
            
            response = self.s3.get_object(Bucket=self.artifact_bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            return content
        except Exception as e:
            error_msg = f"Error getting artifact content: {str(e)}"
            return error_msg
    
    def parse_artifact_content(self, key: str, content: str) -> Any:
        """
        Parse artifact content based on file extension.
        
        Args:
            key: S3 key of the artifact
            content: Content of the artifact
            
        Returns:
            Any: Parsed content
        """
        try:
            if key.endswith('.json'):
                return json.loads(content)
            elif key.endswith('.yaml') or key.endswith('.yml'):
                return yaml.safe_load(content)
            else:
                return content
        except Exception as e:
            st.warning(f"Could not parse artifact content: {str(e)}")
            return content
    
    def render_migration_summary(self, df: pd.DataFrame) -> None:
        """
        Render migration summary statistics.
        
        Args:
            df: DataFrame containing migration data
        """
        # Check if DataFrame is empty
        if df.empty:
            st.info("No migration data available. Upload migration input file to Artifacts bucket and use the top button to begin migration.")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Services", 0)
            
            with col2:
                st.metric("Artifacts Generated", 0)
                
            with col3:
                st.metric("Pending", 0)
            
            with col4:
                st.metric("In Progress", 0)
            
            with col5:
                st.metric("Failed", 0)
            return
            
        # Count by status
        if 'status' in df.columns:
            # Convert DataFrame to ensure we're working with clean data
            df_clean = df.copy()
            
            # Force status to be string type
            if 'status' in df_clean.columns:
                df_clean['status'] = df_clean['status'].astype(str)
            
            # Get value counts
            status_counts = df_clean['status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                # Convert to int to avoid float display
                total_services = int(len(df_clean))
                st.metric("Total Services", total_services)
            
            with col2:
                completed = int(status_counts[status_counts['Status'] == 'COMPLETED']['Count'].sum() if not status_counts[status_counts['Status'] == 'COMPLETED'].empty else 0)
                st.metric("Artifacts Generated", completed)
            
            with col3:
                pending = int(status_counts[status_counts['Status'] == 'PENDING']['Count'].sum() if not status_counts[status_counts['Status'] == 'PENDING'].empty else 0)
                st.metric("Pending", pending)
            
            with col4:
                in_progress = int(status_counts[status_counts['Status'] == 'IN_PROGRESS']['Count'].sum() if not status_counts[status_counts['Status'] == 'IN_PROGRESS'].empty else 0)
                st.metric("In Progress", in_progress)
            
            with col5:
                failed = int(status_counts[status_counts['Status'] == 'FAILED']['Count'].sum() if not status_counts[status_counts['Status'] == 'FAILED'].empty else 0)
                st.metric("Failed", failed, delta_color="inverse")
        else:
            st.error("Status column not found in data")
    
    def render_migration_list(self, df: pd.DataFrame) -> None:
        """
        Render list of migrations.
        
        Args:
            df: DataFrame containing migration data
        """
        st.header("Migration List")
        
        # Add status filter for the migration list
        unique_statuses = sorted(df['status'].unique().tolist())
        
        # Ensure all standard statuses are included, even if no migrations currently have that status
        standard_statuses = ['PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED']
        for status in standard_statuses:
            if status not in unique_statuses:
                unique_statuses.append(status)
        
        # Sort the combined list and add 'All' at the beginning
        status_values = ['All'] + sorted(unique_statuses)
        selected_status = st.selectbox(
            "Filter by Status",
            status_values,
            key="list_status_filter"
        )
        
        # Add Process Pending Migrations button when PENDING is selected
        if selected_status == 'PENDING':
            # Use a more prominent button with custom styling
            if st.button("Process Pending Migrations", type="primary", use_container_width=True):
                # Create a placeholder for the status message
                status_placeholder = st.empty()
                
                # Show a more prominent spinner
                with status_placeholder.container():
                    st.markdown("### Processing pending migrations...")
                    st.markdown("Please wait while the system processes pending migrations. This may take a few moments.")
                    progress_bar = st.progress(0)
                    for i in range(100):
                        # Simulate progress for visual feedback
                        progress_bar.progress(i + 1)
                        time.sleep(0.01)
                
                # Process the migrations
                success = self.process_pending_migrations()
                
                # Clear the spinner and show a more prominent result
                status_placeholder.empty()
                if success:
                    # Show a prominent success message
                    with status_placeholder.container():
                        st.markdown("### ✅ Success!")
                        st.markdown("**Pending migrations have been successfully triggered for processing.**")
                        st.markdown("The system will start processing them based on available capacity.")
                        st.markdown("You can refresh this page in a few moments to see updated status.")
                    
                    # Clear cache to force reload on next refresh
                    if 'migration_data_cache' in st.session_state:
                        del st.session_state.migration_data_cache
                    
                    # Add a delay before rerunning to ensure the message is seen
                    time.sleep(5)
                    st.rerun()
                else:
                    # Show a prominent error message
                    with status_placeholder.container():
                        st.markdown("### ❌ Error")
                        st.markdown("**Failed to process pending migrations.**")
                        st.markdown("Please check the logs for details or try again later.")
        
        # Apply status filter if not 'All'
        if selected_status != 'All':
            filtered_df = df[df['status'] == selected_status]
        else:
            filtered_df = df
            
        # Store the filtered dataframe in session state for access in other parts of the app
        st.session_state.filtered_df = filtered_df
        st.session_state.list_selected_status = selected_status
        # Prepare data for display with error handling
        try:
            # If no migrations match the filter, create an empty display dataframe but continue
            if filtered_df.empty:
                # Create an empty display dataframe with the required columns
                display_df = pd.DataFrame(columns=['service_name', 'status', 'current_stage', 'created_at', 'updated_at'])
                # We'll show a message in the table instead of a separate info box
            else:
                # Check for required columns and create a display dataframe
                display_df = filtered_df.copy()
                
                # Add missing columns with default values if they don't exist
                required_columns = ['service_name', 'status', 'current_stage', 'created_at', 'updated_at']
                for col in required_columns:
                    if col not in display_df.columns:
                        display_df[col] = "N/A"
                
                # Ensure all columns are of the correct type
                if 'status' in display_df.columns:
                    display_df['status'] = display_df['status'].astype(str)
                if 'current_stage' in display_df.columns:
                    display_df['current_stage'] = display_df['current_stage'].astype(str)
            
            # Select columns for display
            display_df = display_df[required_columns].copy()
            
            # Format datetime columns
            for col in ['created_at', 'updated_at']:
                try:
                    if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                        display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        # Try to convert to datetime if it's not already
                        display_df[col] = pd.to_datetime(display_df[col], errors='coerce')
                        display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    # If conversion fails, ensure it's a string
                    display_df[col] = display_df[col].astype(str)
            
            # Add a visual progress indicator column
            display_df['Progress'] = display_df.apply(self._calculate_progress_indicator, axis=1)
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'service_name': 'Service',
                'status': 'Migration Status',
                'current_stage': 'Current Stage',
                'created_at': 'Created',
                'updated_at': 'Last Updated'
            })
            
            # Display the dataframe with the progress indicator or a message if empty
            if display_df.empty:
                st.info(f"No migrations with status '{selected_status}' found.")
            else:
                st.dataframe(display_df, hide_index=True)
            
        except Exception as e:
            st.error(f"Error preparing display data: {str(e)}")
            st.code(traceback.format_exc())
    
    def _calculate_progress_indicator(self, row):
        """
        Calculate a progress indicator for a migration row.
        
        Args:
            row: DataFrame row
            
        Returns:
            str: Progress indicator string
        """
        try:
            # Define the migration stages in order
            migration_stages = [
                'CONTAINER_CONFIG', 
                'VALIDATION', 
                'SECURITY', 
                'DOCUMENTATION',
                'COMPLETED'  # Added COMPLETED as a final stage
            ]
            
            # Get the current stage and status
            current_stage = str(row.get('current_stage', 'INIT'))
            status = str(row.get('status', 'PENDING'))
            
            # Special case for COMPLETED status
            if status == 'COMPLETED' or current_stage == 'COMPLETED':
                return "100%"
                
            # Handle INIT stage specially
            if current_stage == 'INIT':
                if status == 'PENDING':
                    return "Queued (0%)"
                elif status == 'IN_PROGRESS':
                    return "Queued (0%)"
                else:
                    return f"{status} (0%)"
            
            # Calculate progress percentage
            if status == 'COMPLETED':
                progress_percentage = 100
            elif status == 'FAILED':
                # Find the index of the current stage
                try:
                    if current_stage in migration_stages:
                        stage_index = migration_stages.index(current_stage)
                        progress_percentage = int((stage_index / (len(migration_stages) - 1)) * 100)
                    else:
                        progress_percentage = 0
                except Exception:
                    progress_percentage = 0
            else:  # IN_PROGRESS
                try:
                    if current_stage in migration_stages:
                        stage_index = migration_stages.index(current_stage)
                        # For in-progress migrations, calculate percentage based on stage position
                        progress_percentage = int((stage_index / len(migration_stages)) * 100)
                    else:
                        progress_percentage = 0
                except Exception:
                    progress_percentage = 0
            
            # Return a progress indicator string
            return f"{progress_percentage}%"
        except Exception as e:
            # If any error occurs, return a default value
            return "0%"
    
    def render_migration_details(self, df: pd.DataFrame) -> None:
        """
        Render detailed view of a selected migration.
        
        Args:
            df: DataFrame containing migration data
        """
        st.header("Migration Details")
        
        # Check if DataFrame is empty
        if df.empty:
            st.info("No migration data available. Upload migration input file to Artifacts bucket and use the top button to begin migration.")
            return
        
        try:
            # Force refresh the data to ensure we have the latest status
            st.session_state.force_refresh = True
            df = self.load_migration_data()
            
            # Check if required columns exist
            if 'migration_id' not in df.columns:
                st.warning("Required column 'migration_id' not found in data. The migration table may be corrupted.")
                return
            # Store the filtered dataframe in session state for access in other parts of the app
            st.session_state.filtered_df = df
            
            # Create a dictionary for selection
            # Use migration_id as both key and display value if service_name is not available
            if 'service_name' in df.columns:
                service_to_id = dict(zip(df['service_name'], df['migration_id']))
                
                # Natural sort for service names (to handle numeric suffixes correctly)
                import re
                def natural_sort_key(s):
                    # Split the string into text and numeric parts
                    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
                
                service_names = sorted(service_to_id.keys(), key=natural_sort_key)
            else:
                # Use migration_id for both display and selection
                service_to_id = dict(zip(df['migration_id'], df['migration_id']))
                service_names = sorted(service_to_id.keys())
                st.warning("Service name column not found, using migration IDs instead")
            
            # Get unique status values for filtering
            unique_statuses = sorted(df['status'].unique().tolist())
            
            # Ensure all standard statuses are included, even if no migrations currently have that status
            standard_statuses = ['PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED']
            for status in standard_statuses:
                if status not in unique_statuses:
                    unique_statuses.append(status)
            
            # Sort the combined list and add 'All' at the beginning
            status_values = ['All'] + sorted(unique_statuses)
            
            # Create columns for filters
            filter_col1, filter_col2 = st.columns([2, 1])
            
            with filter_col2:
                # Status filter dropdown
                selected_status = st.selectbox(
                    "Filter by Status",
                    status_values,
                    key="status_filter"
                )
                
                # Add Process Pending Migrations button when PENDING is selected
                if selected_status == 'PENDING':
                    # Use a more prominent button with custom styling
                    if st.button("Process Pending Migrations", type="primary", key="detail_process_pending", use_container_width=True):
                        # Create a placeholder for the status message
                        status_placeholder = st.empty()
                        
                        # Show a more prominent spinner
                        with status_placeholder.container():
                            st.markdown("### Processing pending migrations...")
                            st.markdown("Please wait while the system processes pending migrations. This may take a few moments.")
                            progress_bar = st.progress(0)
                            for i in range(100):
                                # Simulate progress for visual feedback
                                progress_bar.progress(i + 1)
                                time.sleep(0.01)
                        
                        # Process the migrations
                        success = self.process_pending_migrations()
                        
                        # Clear the spinner and show a more prominent result
                        status_placeholder.empty()
                        if success:
                            # Show a prominent success message
                            with status_placeholder.container():
                                st.markdown("### ✅ Success!")
                                st.markdown("**Pending migrations have been successfully triggered for processing.**")
                                st.markdown("The system will start processing them based on available capacity.")
                                st.markdown("You can refresh this page in a few moments to see updated status.")
                            
                            # Clear cache to force reload on next refresh
                            if 'migration_data_cache' in st.session_state:
                                del st.session_state.migration_data_cache
                            
                            # Add a delay before rerunning to ensure the message is seen
                            time.sleep(5)
                            st.rerun()
                        else:
                            # Show a prominent error message
                            with status_placeholder.container():
                                st.markdown("### ❌ Error")
                                st.markdown("**Failed to process pending migrations.**")
                                st.markdown("Please check the logs for details or try again later.")
                
                # Reset migration selection when status filter changes
                if "previous_status_filter" not in st.session_state:
                    st.session_state.previous_status_filter = selected_status
                elif st.session_state.previous_status_filter != selected_status:
                    # Status filter changed, reset migration selection
                    if 'migration_selection' in st.session_state:
                        del st.session_state.migration_selection
                    st.session_state.previous_status_filter = selected_status
            
            # Apply status filter if not 'All'
            if selected_status != 'All':
                filtered_df = df[df['status'] == selected_status]
            else:
                filtered_df = df
                
            # Store the filtered dataframe in session state for access in other parts of the app
            st.session_state.filtered_df = filtered_df
            st.session_state.detail_selected_status = selected_status
            
            # Create a callback function to update session state when selection changes
            def on_migration_select():
                # This will be called when the selectbox value changes
                if st.session_state.migration_selector != "Select":
                    if st.session_state.migration_selector in service_to_id:
                        selected_migration_id = service_to_id[st.session_state.migration_selector]
                        st.session_state.selected_migration_id = selected_migration_id
                        st.session_state.migration_selection = st.session_state.migration_selector
                else:
                    # Clear selection if "Select" is chosen
                    if 'selected_migration_id' in st.session_state:
                        del st.session_state.selected_migration_id
                    if 'migration_selection' in st.session_state:
                        del st.session_state.migration_selection
            
            with filter_col1:
                # Update service_to_id and service_names based on filtered data
                if not filtered_df.empty:
                    if 'service_name' in filtered_df.columns:
                        service_to_id = dict(zip(filtered_df['service_name'], filtered_df['migration_id']))
                        service_names = sorted(service_to_id.keys(), key=natural_sort_key)
                    else:
                        service_to_id = dict(zip(filtered_df['migration_id'], filtered_df['migration_id']))
                        service_names = sorted(service_to_id.keys())
                    
                    # Add a "Select" option at the beginning of the list
                    service_names_with_select = ["Select"] + service_names
                else:
                    # No migrations match the filter, just show "Select" option
                    service_names_with_select = ["Select"]
                
                # Select service name or migration ID from dropdown - always use "Select Migration" as the label
                selected_item = st.selectbox(
                    "Select Migration", 
                    service_names_with_select,
                    key="migration_selector",
                    on_change=on_migration_select
                )
            
            # Check if we have a valid selection and it's not "Select" and we have migrations
            if selected_item and selected_item != "Select" and not filtered_df.empty:
                # Get the corresponding migration ID
                selected_migration = service_to_id[selected_item]
                
                # Add refresh button for migration progress, timeline, and artifacts
                col1, col2 = st.columns([0.9, 0.1])
                with col2:
                    refresh_button = st.button("🔄", key="refresh_migration_details")
                
                # If refresh button is clicked, clear the cache for this migration
                if refresh_button:
                    # Clear migration data cache to force reload
                    if 'migration_data_cache' in st.session_state:
                        del st.session_state.migration_data_cache
                    
                    # Clear artifacts cache for this migration
                    if 'artifacts_cache' in st.session_state and selected_migration in st.session_state.artifacts_cache:
                        del st.session_state.artifacts_cache[selected_migration]
                    
                    # Reload the data
                    df = self.load_migration_data()
                
                # Get migration details (will be refreshed if button was clicked)
                migration = df[df['migration_id'] == selected_migration].iloc[0]
                
                # Determine if this is a failed migration
                is_failed = migration.get('status') == 'FAILED'
                is_completed = migration.get('status') == 'COMPLETED' or migration.get('current_stage') == 'COMPLETED'
                # Get current stage first to avoid reference before assignment
                current_stage = str(migration.get('current_stage', 'INIT'))
                is_init = migration.get('status') == 'INIT' or current_stage == 'INIT'
                
                # Add a visual progress indicator for this migration
                st.subheader("Migration Progress")
                
                # Add action buttons based on migration status
                col1, col2 = st.columns([1, 1])
                
                # with col1:
                #     if is_failed:
                #         if st.button("🔄 Restart Failed Migration", type="primary", use_container_width=True):
                #             # Create a placeholder for the status message
                #             status_placeholder = st.empty()
                            
                #             # Show a more prominent spinner
                #             with status_placeholder.container():
                #                 st.markdown("### Restarting migration...")
                #                 st.markdown("Please wait while the system restarts the failed migration. This may take a few moments.")
                #                 progress_bar = st.progress(0)
                #                 for i in range(100):
                #                     # Simulate progress for visual feedback
                #                     progress_bar.progress(i + 1)
                #                     time.sleep(0.01)
                            
                #             # Restart the migration
                #             success = self.restart_migration(selected_migration)
                            
                #             # Clear the spinner and show a more prominent result
                #             status_placeholder.empty()
                #             if success:
                #                 # Show a prominent success message
                #                 with status_placeholder.container():
                #                     st.markdown("### ✅ Success!")
                #                     st.markdown("**Migration restarted successfully!**")
                #                     st.markdown("The system will begin processing the migration shortly.")
                #                     st.markdown("You can refresh this page in a few moments to see updated status.")
                                
                #                 # Clear cache to force reload on next refresh
                #                 if 'migration_data_cache' in st.session_state:
                #                     del st.session_state.migration_data_cache
                                
                #                 # Add a delay before rerunning to ensure the message is seen
                #                 time.sleep(5)
                #                 st.rerun()
                #             else:
                #                 # Show a prominent error message
                #                 with status_placeholder.container():
                #                     st.markdown("### ❌ Error")
                #                     st.markdown("**Failed to restart migration.**")
                #                     st.markdown("Please check the logs for details or try again later.")
                
                # Define the migration stages in order
                migration_stages = [
                    'CONTAINER_CONFIG', 
                    'VALIDATION', 
                    'SECURITY', 
                    'DOCUMENTATION',
                    'COMPLETED'  # Added COMPLETED as a final stage
                ]
                
                # Get the current stage
                current_stage = str(migration.get('current_stage', 'CONTAINER_CONFIG'))
                current_status = str(migration.get('status', 'PENDING'))
                
                # Calculate progress percentage
                if current_status == 'COMPLETED' or current_stage == 'COMPLETED':
                    progress_percentage = 100
                elif current_status == 'FAILED':
                    # Find the index of the current stage
                    try:
                        if current_stage in migration_stages:
                            stage_index = migration_stages.index(current_stage)
                            progress_percentage = int((stage_index / len(migration_stages)) * 100)
                        else:
                            # If stage not found in migration_stages, default to 0%
                            progress_percentage = 0
                    except Exception:
                        progress_percentage = 0
                else:  # IN_PROGRESS
                    try:
                        # For INIT stage with IN_PROGRESS status, show as queued
                        if current_stage == 'INIT':
                            progress_percentage = 0
                        elif current_stage in migration_stages:
                            stage_index = migration_stages.index(current_stage)
                            # For in-progress migrations, calculate percentage based on stage position
                            progress_percentage = int((stage_index / len(migration_stages)) * 100)
                        else:
                            # If stage not found in migration_stages, default to 0%
                            progress_percentage = 0
                    except Exception:
                        progress_percentage = 0
                        # For the current stage, we show it as in progress (not complete)
                        progress_percentage = int((stage_index / len(migration_stages)) * 100)
                    except ValueError:
                        progress_percentage = 0
                
                # Display progress bar
                if current_status == 'COMPLETED':
                    st.progress(1.0)
                    st.success(f"Migration Artifacts generated successfully (100%)")
                elif current_status == 'FAILED':
                    st.progress(progress_percentage / 100)
                    
                    # Define display names for stages
                    stage_display_names = {
                        'CONTAINER_CONFIG': 'Container Configuration',
                        'VALIDATION': 'Validation & Test Cases',
                        'SECURITY': 'Security',
                        'DOCUMENTATION': 'Documentation',
                        'INIT': 'Initialization',
                        'COMPLETED': 'Completed'
                    }
                    
                    display_stage = stage_display_names.get(current_stage, current_stage)
                    st.error(f"Artifact generation failed at {display_stage} stage ({progress_percentage}%)")
                else:  # IN_PROGRESS
                    st.progress(progress_percentage / 100)
                    
                    # Define display names for stages
                    stage_display_names = {
                        'CONTAINER_CONFIG': 'Container Configuration',
                        'VALIDATION': 'Validation & Test Cases',
                        'SECURITY': 'Security',
                        'DOCUMENTATION': 'Documentation',
                        'INIT': 'Initialization',
                        'COMPLETED': 'Completed'
                    }
                    
                    display_stage = stage_display_names.get(current_stage, current_stage)
                    
                    # Special handling for INIT stage
                    if current_stage == 'INIT':
                        st.info(f"Migration queued and waiting to start (0%)")
                    else:
                        st.info(f"Generating {display_stage} artifacts ({progress_percentage}%)")
                
                # Display stage timeline
                st.subheader("Migration Timeline")

                # Handle INIT stage specially
                if current_stage == 'INIT':
                    if current_status == 'IN_PROGRESS' or current_status == 'PENDING':
                        st.info("This migration is queued and waiting to start. It will begin processing when resources are available.")
                    else:
                        st.error(f"Migration failed during initialization: {current_status}")
                elif current_stage == 'COMPLETED' and current_status == 'COMPLETED':
                    # Create columns for each stage
                    timeline_cols = st.columns(len(migration_stages))
                    
                    # Define display names for stages
                    stage_display_names = {
                        'CONTAINER_CONFIG': 'Container Configuration',
                        'VALIDATION': 'Validation & Test Cases',
                        'SECURITY': 'Security',
                        'DOCUMENTATION': 'Documentation',
                        'COMPLETED': 'Completed'
                    }
                    
                    # Display each stage with appropriate styling - all completed
                    for i, stage in enumerate(migration_stages):
                        with timeline_cols[i]:
                            st.markdown(f"**{stage_display_names[stage]}**")
                            st.markdown(f"<div style='background-color:#0068C9; color:white; padding:5px; border-radius:5px; text-align:center; font-size:small;'>✓</div>", unsafe_allow_html=True)
                else:
                    # Create columns for each stage
                    timeline_cols = st.columns(len(migration_stages))
                    
                    # Define display names for stages
                    stage_display_names = {
                        'CONTAINER_CONFIG': 'Container Configuration',
                        'VALIDATION': 'Validation & Test Cases',
                        'SECURITY': 'Security',
                        'DOCUMENTATION': 'Documentation',
                        'COMPLETED': 'Completed'
                    }
                    
                    # Display each stage with appropriate styling
                    for i, stage in enumerate(migration_stages):
                        with timeline_cols[i]:
                            st.markdown(f"**{stage_display_names[stage]}**")
                            
                            # Determine the stage status
                            if current_status == 'COMPLETED':
                                # All stages are complete
                                st.markdown(f"<div style='background-color:#0068C9; color:white; padding:5px; border-radius:5px; text-align:center; font-size:small;'>✓</div>", unsafe_allow_html=True)
                            elif current_status == 'FAILED' and stage == current_stage:
                                # This is the failed stage
                                st.markdown(f"<div style='background-color:#FF2B2B; color:white; padding:5px; border-radius:5px; text-align:center; font-size:small;'>✗</div>", unsafe_allow_html=True)
                            elif current_status == 'IN_PROGRESS' and stage == current_stage:
                                # This is the current stage
                                st.markdown(f"<div style='background-color:#83C9FF; color:white; padding:5px; border-radius:5px; text-align:center; font-size:small;'>⟳</div>", unsafe_allow_html=True)
                            elif migration_stages.index(stage) < migration_stages.index(current_stage):
                                # This stage is complete
                                st.markdown(f"<div style='background-color:#0068C9; color:white; padding:5px; border-radius:5px; text-align:center; font-size:small;'>✓</div>", unsafe_allow_html=True)
                            else:
                                # This stage is pending
                                st.markdown(f"<div style='background-color:#f0f2f6; padding:5px; border-radius:5px; text-align:center; font-size:small;'>-</div>", unsafe_allow_html=True)
                
                # Display service information in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Service Information")
                    if 'service_name' in migration:
                        st.write(f"**Service Name:** {migration.get('service_name')}")
                    
                    # Define display names for stages
                    stage_display_names = {
                        'CONTAINER_CONFIG': 'Container Configuration',
                        'VALIDATION': 'Validation & Test Cases',
                        'SECURITY': 'Security',
                        'DOCUMENTATION': 'Documentation'
                    }
                    
                    current_stage = migration.get('current_stage', 'N/A')
                    display_stage = stage_display_names.get(current_stage, current_stage)
                    st.write(f"**Current Stage:** {display_stage}")
                    
                    # Format timestamps using the helper function
                    created_at = format_timestamp(migration.get('created_at', 'N/A'))
                    updated_at = format_timestamp(migration.get('updated_at', 'N/A'))
                    
                    st.write(f"**Created At:** {created_at}")
                    st.write(f"**Updated At:** {updated_at}")
                    st.write(f"**Migration ID:** {selected_migration}")
                
                with col2:
                    st.subheader("Service Details")
                    if 'service_info' in migration:
                        service_info = migration['service_info']
                        st.write(f"**Description:** {service_info.get('description', 'N/A')}")
                        st.write(f"**Repository:** {service_info.get('repository_url', 'N/A')}")
                        st.write(f"**Project Name:** {service_info.get('project_name', 'N/A')}")
                        st.write(f"**Version:** {service_info.get('current_version', 'N/A')}")
                        st.write(f"**Helm Chart:** {service_info.get('helm_chart_description', 'N/A')}")
                        st.write(f"**Number of Apps:** {service_info.get('number_of_apps', 'N/A')}")
                    else:
                        st.warning("No service information available")
                
                with col3:
                    st.subheader("Configuration")
                    if 'service_info' in migration:
                        service_info = migration['service_info']
                        st.write(f"**CPU Limits:** {service_info.get('cpu_limits', 'N/A')}")
                        st.write(f"**Memory Limits:** {service_info.get('memory_limits', 'N/A')}")
                        st.write(f"**Dependencies:** {service_info.get('additional_dependencies', 'None')}")
                    else:
                        st.warning("No configuration information available")
                
                # Display Apps Configuration if available
                if 'service_info' in migration and 'apps_config' in migration['service_info']:
                    st.subheader("Application Configurations")
                    apps_config = migration['service_info']['apps_config']
                    
                    if apps_config:
                        for i, app in enumerate(apps_config):
                            with st.expander(f"📱 {app.get('name', f'App {i+1}')} - {app.get('version', 'N/A')}"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write(f"**Description:** {app.get('description', 'N/A')}")
                                    st.write(f"**Version:** {app.get('version', 'N/A')}")
                                    st.write(f"**Ingress:** {'✅ Yes' if app.get('ingress', False) else '❌ No'}")
                                    if app.get('ingress', False):
                                        st.write(f"**Ingress Type:** {app.get('ingress_type', 'N/A')}")
                                        st.write(f"**Hostname:** {app.get('hostname', 'N/A')}")
                                    st.write(f"**Health Check:** {app.get('healthcheck_path', 'N/A')}")
                                
                                with col2:
                                    st.write(f"**CPU Limits:** {app.get('cpu_limits', 'N/A')}")
                                    st.write(f"**Memory Limits:** {app.get('memory_limits', 'N/A')}")
                                    st.write(f"**Kafka Consumer:** {'✅ Yes' if app.get('kafka_consumer', False) else '❌ No'}")
                                    st.write(f"**Service Monitor:** {'✅ Yes' if app.get('service_monitor', False) else '❌ No'}")
                                    st.write(f"**StatsD Exporter:** {'✅ Yes' if app.get('statsd_exporter', False) else '❌ No'}")
                                
                                with col3:
                                    st.write(f"**AWS Permissions:** {'✅ Yes' if app.get('aws_permissions', False) else '❌ No'}")
                                    st.write(f"**Custom Role:** {'✅ Yes' if app.get('custom_role', False) else '❌ No'}")
                                    st.write(f"**Empty Volumes:** {'✅ Yes' if app.get('empty_volumes', False) else '❌ No'}")
                                    st.write(f"**Canary Support:** {'✅ Yes' if app.get('canary_support', False) else '❌ No'}")
                    else:
                        st.info("No application configurations found")
                
                # Display error details if this is a failed migration
                if 'error_details' in migration:
                    # Only show error details if the status is actually FAILED
                    if migration.get('status') == 'FAILED':
                        st.header("⚠️ Failure Details")
                        
                        try:
                            error_details = json.loads(migration['error_details'])
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.error(f"**Failed at Agent:** {error_details.get('agent', 'Unknown').title()}")
                                st.error(f"**Error Code:** {error_details.get('error_code', 'Unknown')}")
                            
                            with col2:
                                st.error(f"**Error Message:** {error_details.get('message', 'Unknown error occurred')}")
                            
                            # Show recommendations if available
                            if 'recommendations' in error_details:
                                st.subheader("Recommendations")
                                for rec in error_details['recommendations']:
                                    st.info(f"• {rec}")
                                    
                            # Add general troubleshooting guidance
                            st.subheader("Troubleshooting Steps")
                            
                            # Get the agent name for the CloudWatch logs command
                            agent_name = error_details.get('agent', '').lower()
                            if not agent_name:
                                agent_name = "UNKNOWN"  # Default to UNKNOWN if not specified
                                
                            st.markdown(f"""
                            ### General Troubleshooting Steps
                            
                            1. **View CloudWatch Logs**: Check the Lambda logs for detailed error messages
                              
                            2. **Retry the migration**:
                               - Click the "START NEW MIGRATION" button at the top of this page
                               - This will restart the migration workflow and retry the failed step
                            
                            3. **If throttling occurred**: Wait a few minutes before retrying to allow rate limits to reset
                            """)
                        except:
                            st.error(f"**Error Details:** {migration['error_details']}")
                            
                            # Add troubleshooting guidance even if JSON parsing fails
                            st.subheader("Troubleshooting Steps")
                            
                            st.markdown("""
                            ### General Troubleshooting Steps
                            
                            1. **View CloudWatch Logs**: Check the Lambda logs for detailed error messages:
                              
                            2. **Retry the migration**:
                               - Click the "START NEW MIGRATION" button at the top of this page
                               - This will restart the migration workflow and retry the failed step
                            
                            3. **If throttling occurred**: Wait a few minutes before retrying to allow rate limits to reset
                            """)
                
                # Load artifacts
                artifacts = self.load_artifacts(selected_migration)
                
                if artifacts:
                    st.header("Artifacts")
                    
                    # Create tabs for different artifact types
                    tab_names = ["Container Config", "Validation & Test Cases", "Documentation", "GitLab", "Bedrock Assistant"]
                    
                    # For Amazon Q Assistant, we don't need artifacts to show it
                    available_tabs = []
                    for tab in tab_names:
                        if tab == "Bedrock Assistant":
                            available_tabs.append(tab)
                        elif tab == "Validation & Test Cases" and "validation" in artifacts:
                            available_tabs.append(tab)
                        elif tab.lower().replace(" ", "_") in artifacts:
                            available_tabs.append(tab)
                    
                    if not available_tabs:
                        st.info("No artifacts found for this migration.")
                        return
                    
                    # Store the selected tab in session state if it doesn't exist
                    if 'selected_artifact_tab' not in st.session_state:
                        st.session_state.selected_artifact_tab = "Container Config"
                    
                    # Use radio buttons to select the tab
                    selected_tab = st.radio(
                        "Select Tab", 
                        available_tabs, 
                        horizontal=True, 
                        label_visibility="collapsed",
                        key="artifact_tab_selector_" + selected_migration,  # Add migration ID to make the key unique
                        index=available_tabs.index(st.session_state.selected_artifact_tab) if st.session_state.selected_artifact_tab in available_tabs else 0
                    )
                    
                    # Only update the session state if the tab has actually changed
                    if selected_tab != st.session_state.selected_artifact_tab:
                        st.session_state.selected_artifact_tab = selected_tab
                        # Force a rerun to apply the tab change immediately
                        st.rerun()
                    
                    # Display artifacts based on selected tab
                    if selected_tab == "Container Config":
                        self._render_container_config_artifacts(artifacts.get('container_config', []), selected_migration)
                    elif selected_tab == "Validation & Test Cases":
                        self._render_validation_artifacts(artifacts.get('validation', []), selected_migration)
                    elif selected_tab == "Documentation":
                        self._render_documentation_artifacts(artifacts.get('documentation', []), selected_migration)
                    elif selected_tab == "GitLab":
                        self._render_gitlab_artifacts(artifacts.get('gitlab', []), selected_migration)
                    elif selected_tab == "Bedrock Assistant":
                        self._render_q_assistant(selected_migration, artifacts)
                else:
                    st.info("No artifacts found for this migration.")
            else:
                # Show different message based on whether there are no migrations matching the filter
                # or user just hasn't selected a migration yet
                if 'filtered_df' in st.session_state and st.session_state.filtered_df.empty:
                    # Get the status from the appropriate session state variable
                    if 'detail_selected_status' in st.session_state:
                        status = st.session_state.detail_selected_status
                    elif 'list_selected_status' in st.session_state:
                        status = st.session_state.list_selected_status
                    else:
                        status = "selected"
                    st.info(f"No migrations with status '{status}' found.")
                else:
                    st.info("Please select a migration to view details.")
        except Exception as e:
            st.error(f"Error displaying migration details: {str(e)}")
    
    def _render_container_config_artifacts(self, artifacts: List[Dict], migration_id: str) -> None:
        """
        Render Container Config artifacts.
        
        Args:
            artifacts: List of artifacts
            migration_id: Migration ID
        """
        if not artifacts:
            st.info("No Container Config artifacts found.")
            return
        
        # Separate single-app and multi-app artifacts
        single_app_artifacts = []
        multi_app_artifacts = {}
        umbrella_artifacts = []
        
        for artifact in artifacts:
            name = artifact['name']
            if name.startswith('base/charts/'):
                # Multi-app subchart: base/charts/app_name/...
                parts = name.split('/')
                if len(parts) >= 3:
                    app_name = parts[2]
                    if app_name not in multi_app_artifacts:
                        multi_app_artifacts[app_name] = []
                    multi_app_artifacts[app_name].append({
                        **artifact,
                        'name': '/'.join(parts[3:]) if len(parts) > 3 else 'Chart.yaml'
                    })
            elif name.startswith('base/'):
                # Umbrella chart files: base/Chart.yaml, base/values.yaml
                umbrella_artifacts.append({
                    **artifact,
                    'name': name[5:]  # Remove 'base/' prefix
                })
            else:
                # Single app artifacts
                single_app_artifacts.append(artifact)
        
        # Create appropriate tabs based on structure
        if multi_app_artifacts:
            # Multi-app structure
            tab_names = ["Umbrella Chart"] + list(multi_app_artifacts.keys())
            tabs = st.tabs(tab_names)
            
            # Umbrella Chart tab
            with tabs[0]:
                st.subheader("Umbrella Chart (Multi-App)")
                # Debug: Show what umbrella artifacts we found
                st.write(f"Debug: Found {len(umbrella_artifacts)} umbrella artifacts")
                if umbrella_artifacts:
                    for artifact in umbrella_artifacts:
                        st.write(f"Debug: Umbrella artifact - {artifact['name']}")
                        with st.expander(f"📄 {artifact['name']}"):
                            content = self.get_artifact_content(artifact['key'])
                            if artifact['name'].endswith('.yaml') or artifact['name'].endswith('.yml'):
                                st.code(content, language="yaml")
                            else:
                                st.code(content)
                else:
                    st.info("No umbrella chart files found.")
                    # Debug: Show all artifacts to see what we're working with
                    st.write("Debug: All artifacts:")
                    for artifact in artifacts:
                        st.write(f"- {artifact['name']}")
            
            # Individual app tabs
            for i, (app_name, app_artifacts) in enumerate(multi_app_artifacts.items(), 1):
                with tabs[i]:
                    st.subheader(f"Application: {app_name}")
                    self._render_app_artifacts(app_artifacts, app_name)
        else:
            # Single app structure - use simplified layout
            subtabs = st.tabs(["Dockerfile", "Kubernetes Artifacts", "Helm Template"])
            
            # Dockerfile subtab
            with subtabs[0]:
                dockerfile_found = False
                for artifact in single_app_artifacts:
                    if artifact['name'] == 'Dockerfile':
                        dockerfile_found = True
                        st.subheader("Dockerfile")
                        content = self.get_artifact_content(artifact['key'])
                        st.code(content, language="dockerfile")
                        break
                
                if not dockerfile_found:
                    st.info("No Dockerfile found for this application.")
            
            # Kubernetes Artifacts subtab
            with subtabs[1]:
                chart_files = []
                for artifact in single_app_artifacts:
                    if artifact['name'].endswith('.yaml') and artifact['name'] != 'values.yaml' and not artifact['name'].startswith('values-') and not artifact['name'].startswith('flux-'):
                        chart_files.append(artifact)
                
                if chart_files:
                    for artifact in chart_files:
                        st.subheader(artifact['name'])
                        content = self.get_artifact_content(artifact['key'])
                        st.code(content, language="yaml")
                else:
                    st.info("No Kubernetes artifacts found.")
            
            # Helm Template subtab
            with subtabs[2]:
                st.subheader("Helm Template Output")
                
                if st.button("Generate Helm Template", key="helm_template_btn"):
                    with st.spinner("Running helm template command..."):
                        try:
                            # Find the helm chart directory in S3
                            helm_output = self._run_helm_template(migration_id, single_app_artifacts)
                            if helm_output and not helm_output.startswith("Error:") and not helm_output.startswith("Helm command failed:"):
                                st.success("Helm template generated successfully!")
                                st.code(helm_output, language="yaml")
                            else:
                                st.error(helm_output or "Failed to generate helm template")
                        except Exception as e:
                            st.error(f"Error running helm template: {str(e)}")
                else:
                    st.info("Click 'Generate Helm Template' to run helm template command")
    
    def _run_helm_template(self, migration_id: str, artifacts: List[Dict]) -> str:
        """
        Download S3 artifacts to temp directory and run helm template command.
        
        Args:
            migration_id: Migration ID
            artifacts: List of artifacts
            
        Returns:
            Helm template output as string
        """
        import tempfile
        import subprocess
        import os
        
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Find service name from artifacts
                service_name = None
                for artifact in artifacts:
                    if '/' in artifact['name'] and not artifact['name'].startswith('../'):
                        service_name = artifact['name'].split('/')[0]
                        break
                
                if not service_name:
                    return "Error: Could not determine service name from artifacts"
                
                # Create service directory structure
                service_dir = os.path.join(temp_dir, service_name)
                os.makedirs(service_dir, exist_ok=True)
                
                # Download all artifacts to service directory
                downloaded_files = []
                for artifact in artifacts:
                    try:
                        content = self.get_artifact_content(artifact['key'])
                        
                        # Handle nested service directory structure
                        artifact_name = artifact['name']
                        if artifact_name.startswith(f"{service_name}/"):
                            relative_path = artifact_name[len(service_name)+1:]
                        else:
                            relative_path = artifact_name
                        
                        file_path = os.path.join(service_dir, relative_path)
                        
                        # Create subdirectories if needed
                        dir_path = os.path.dirname(file_path)
                        if dir_path != service_dir:
                            os.makedirs(dir_path, exist_ok=True)
                        
                        with open(file_path, 'w') as f:
                            f.write(content)
                        downloaded_files.append(relative_path)
                    except Exception:
                        continue
                
                # Check if values files exist
                values_file = None
                for file in downloaded_files:
                    if file == "values-flux.yaml":
                        values_file = "values-flux.yaml"
                        break
                    elif file == "values.yaml":
                        values_file = "values.yaml"
                
                if not values_file:
                    return "Error: No values file found (values-flux.yaml or values.yaml)"
                
                # Run helm template command in the service directory
                cmd = ["helm", "template", ".", "--values", values_file, "--debug"]
                result = subprocess.run(
                    cmd,
                    cwd=service_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    return result.stdout
                else:
                    return f"Helm command failed:\n{result.stderr}"
                    
        except subprocess.TimeoutExpired:
            return "Error: Helm command timed out"
        except FileNotFoundError:
            return "Error: Helm command not found. Please ensure Helm is installed."
        except Exception as e:
            return f"Error: {str(e)}"
    
        """Render artifacts for a single app in multi-app structure."""
        if not artifacts:
            st.info(f"No artifacts found for {app_name}.")
            return
        
        # Organize artifacts by type
        chart_files = []
        values_files = []
        template_files = []
        
        for artifact in artifacts:
            name = artifact['name']
            if name == 'Chart.yaml':
                chart_files.append(artifact)
            elif name == 'values.yaml':
                values_files.append(artifact)
            elif name.startswith('templates/'):
                template_files.append(artifact)
        
        # Create subtabs for this app
        app_subtabs = st.tabs(["Chart Files", "Templates", "Values"])
        
        with app_subtabs[0]:
            if chart_files:
                for artifact in chart_files:
                    st.subheader(artifact['name'])
                    content = self.get_artifact_content(artifact['key'])
                    st.code(content, language="yaml")
            else:
                st.info("No chart files found.")
        
        with app_subtabs[1]:
            if template_files:
                for artifact in template_files:
                    with st.expander(f"📄 {artifact['name']}"):
                        content = self.get_artifact_content(artifact['key'])
                        st.code(content, language="yaml")
            else:
                st.info("No template files found.")
        
        with app_subtabs[2]:
            if values_files:
                for artifact in values_files:
                    st.subheader(artifact['name'])
                    content = self.get_artifact_content(artifact['key'])
                    st.code(content, language="yaml")
            else:
                st.info("No values files found.")
            
            # Display flux files
            if flux_files:
                st.markdown("""
                ### Flux CD Configuration
                
                These files define the GitOps deployment configuration using Flux CD for continuous delivery to your EKS cluster.
                """)
                
                for artifact in flux_files:
                    st.subheader(artifact['name'])
                    content = self.get_artifact_content(artifact['key'])
                    
                    # Determine language for syntax highlighting
                    if artifact['name'].endswith('.yaml'):
                        language = "yaml"
                    elif artifact['name'].endswith('.json'):
                        language = "json"
                    else:
                        language = None
                        
                    st.code(content, language=language)
            else:
                st.info("No Flux configuration files found. To generate Flux configurations, include 'generate_flux_configs: true' in your service_info when starting the migration.")
    
    def _render_validation_artifacts(self, artifacts: List[Dict], migration_id: str) -> None:
        """
        Render Validation artifacts.
        
        Args:
            artifacts: List of artifacts
            migration_id: Migration ID
        """
        if not artifacts:
            st.info("No Validation artifacts found.")
            return
        
        # Create tabs for Validation Results and Test Cases
        validation_tabs = st.tabs(["Validation Results", "Test Cases"])
        
        # Validation Results Tab
        with validation_tabs[0]:
            # Add validation summary section
            st.subheader("Validation Checks Summary")
            st.markdown("""
            The validation agent performs the following checks on your Kubernetes configurations:
            
            - **Helm Chart Structure**: Validates the structure of Helm charts against best practices
            - **Kubernetes Resource Validation**: Ensures all resources are properly defined
            - **Deployment Configuration**: Checks for proper replica count, strategy, and selectors
            - **Resource Requirements**: Checks for CPU and memory requests/limits
            - **Health Checks**: Validates liveness and readiness probe configurations
            - **Label Consistency**: Ensures consistent labeling across resources
            """)
            
            # Look for validation results
            validation_results = None
            for artifact in artifacts:
                if artifact['name'] == 'validation-results.json':
                    content = self.get_artifact_content(artifact['key'])
                    validation_results = json.loads(content)
                    break
            
            if validation_results:
                # Display validation status
                if validation_results.get('valid', False):
                    st.success("✅ Validation Passed")
                else:
                    st.error("❌ Validation Failed")
                
                # Display issues count
                issues_count = validation_results.get('issues_count', 0)
                if issues_count > 0:
                    st.warning(f"Found {issues_count} issues")
                else:
                    st.success("No issues found")
                
                # Display issues
                if 'issues' in validation_results and validation_results['issues']:
                    st.subheader("Issues")
                    for issue in validation_results['issues']:
                        st.error(f"• {issue}")
                
                # Display detailed results
                if 'details' in validation_results:
                    st.subheader("Validation Details")
                    
                    # Create expandable sections for each file
                    for file_name, file_results in validation_results['details'].items():
                        with st.expander(f"{file_name} - {'✅ Valid' if file_results.get('valid', False) else '❌ Invalid'}"):
                            # Display issues
                            if 'issues' in file_results and file_results['issues']:
                                st.error("Issues:")
                                for issue in file_results['issues']:
                                    st.error(f"• {issue}")
                            
                            # Display warnings
                            if 'warnings' in file_results and file_results['warnings']:
                                st.warning("Warnings:")
                                for warning in file_results['warnings']:
                                    st.warning(f"• {warning}")
            
            # Look for recommendations
            for artifact in artifacts:
                if artifact['name'] == 'recommendations.json':
                    content = self.get_artifact_content(artifact['key'])
                    recommendations = json.loads(content)
                    
                    st.subheader("Recommendations")
                    for rec in recommendations:
                        st.info(f"• {rec}")
                    break
                    
            # Display validation artifacts in an expandable section
            with st.expander("All Validation Artifacts"):
                for artifact in artifacts:
                    # Only show validation-related artifacts in this tab, not test cases
                    if artifact['name'] != 'test-cases.json':
                        st.subheader(artifact['name'])
                        content = self.get_artifact_content(artifact['key'])
                        try:
                            parsed_content = self.parse_artifact_content(artifact['key'], content)
                            if isinstance(parsed_content, dict) or isinstance(parsed_content, list):
                                st.json(parsed_content)
                            else:
                                st.code(content)
                        except:
                            st.code(content)
        
        # Test Cases Tab
        with validation_tabs[1]:
            # Add test cases summary section
            st.subheader("Test Cases Summary")
            st.markdown("""
            The validation agent generates the following types of test cases:
            
            - **Unit Tests**: Tests for individual components and functions
            - **Integration Tests**: Tests for interactions between components
            - **Performance Tests**: Tests for application performance under load
            """)
            
            # Look for test cases
            test_cases = None
            for artifact in artifacts:
                if artifact['name'] == 'test-cases.json':
                    content = self.get_artifact_content(artifact['key'])
                    test_cases = json.loads(content)
                    break
            
            if test_cases:
                st.success("✅ Test Cases Generated Successfully")
                
                # Create tabs for different test types
                test_type_tabs = st.tabs(["Unit Tests", "Performance Tests", "Integration Tests"])
                
                # Unit Tests Tab
                with test_type_tabs[0]:
                    unit_tests = test_cases.get('unit_tests', [])
                    if unit_tests:
                        st.success(f"Generated {len(unit_tests)} unit tests")
                        
                        for i, test in enumerate(unit_tests):
                            with st.expander(f"{i+1}. {test.get('test_name', 'Unnamed Test')}"):
                                st.markdown(f"**Description:** {test.get('description', 'No description')}")
                                st.markdown("**Test Code:**")
                                st.code(test.get('test_code', 'No code available'), language="python")
                                st.markdown(f"**Expected Outcome:** {test.get('expected_outcome', 'Not specified')}")
                                
                                if test.get('setup'):
                                    st.markdown(f"**Setup:** {test.get('setup')}")
                                
                                if test.get('teardown'):
                                    st.markdown(f"**Teardown:** {test.get('teardown')}")
                    else:
                        st.info("No unit tests generated.")
                
                # Performance Tests Tab
                with test_type_tabs[1]:
                    performance_tests = test_cases.get('performance_tests', [])
                    if performance_tests:
                        st.success(f"Generated {len(performance_tests)} performance tests")
                        
                        for i, test in enumerate(performance_tests):
                            with st.expander(f"{i+1}. {test.get('test_name', 'Unnamed Test')}"):
                                st.markdown(f"**Description:** {test.get('description', 'No description')}")
                                st.markdown(f"**Test Tool:** {test.get('test_tool', 'Not specified')}")
                                st.markdown("**Test Script:**")
                                st.code(test.get('test_script', 'No script available'))
                                st.markdown(f"**Success Criteria:** {test.get('success_criteria', 'Not specified')}")
                                st.markdown(f"**Duration:** {test.get('duration', 'Not specified')}")
                                st.markdown(f"**Load Profile:** {test.get('load_profile', 'Not specified')}")
                    else:
                        st.info("No performance tests generated.")
                
                # Integration Tests Tab
                with test_type_tabs[2]:
                    integration_tests = test_cases.get('integration_tests', [])
                    if integration_tests:
                        st.success(f"Generated {len(integration_tests)} integration tests")
                        
                        for i, test in enumerate(integration_tests):
                            with st.expander(f"{i+1}. {test.get('test_name', 'Unnamed Test')}"):
                                st.markdown(f"**Description:** {test.get('description', 'No description')}")
                                st.markdown("**Test Code:**")
                                st.code(test.get('test_code', 'No code available'))
                                st.markdown(f"**Expected Outcome:** {test.get('expected_outcome', 'Not specified')}")
                                
                                if test.get('setup'):
                                    st.markdown(f"**Setup:** {test.get('setup')}")
                                
                                if test.get('teardown'):
                                    st.markdown(f"**Teardown:** {test.get('teardown')}")
                    else:
                        st.info("No integration tests generated.")
                
                # Provide both download button and copy options
                st.subheader("Export Test Cases")
                
                # Format the JSON with indentation for better readability
                test_cases_json = json.dumps(test_cases, indent=2, cls=DecimalEncoder)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download button option
                    st.download_button(
                        label="Download Test Cases JSON",
                        data=test_cases_json,
                        file_name="test-cases.json",
                        mime="application/json",
                        key="download_test_cases"
                    )
                
                # Second column intentionally left empty for layout balance
            else:
                st.info("No test cases found. Test cases are generated asynchronously and may take a few minutes to appear.")
                st.warning("If test cases don't appear after a few minutes, check the logs for any errors in the test case generation process.")
                
            # Display testing artifacts in an expandable section
            with st.expander("All Testing Artifacts"):
                for artifact in artifacts:
                    # Only show test-cases.json in this tab
                    if artifact['name'] == 'test-cases.json':
                        st.subheader(artifact['name'])
                        content = self.get_artifact_content(artifact['key'])
                        try:
                            parsed_content = self.parse_artifact_content(artifact['key'], content)
                            if isinstance(parsed_content, dict) or isinstance(parsed_content, list):
                                st.json(parsed_content)
                            else:
                                st.code(content)
                        except:
                            st.code(content)
        
        # Display validation artifacts in an expandable section (moved inside the validation tab)
        # with st.expander("All Validation Artifacts"):
        #     for artifact in artifacts:
        #         # Only show validation-related artifacts in this tab, not test cases
        #         if artifact['name'] != 'test-cases.json':
        #             st.subheader(artifact['name'])
        #             content = self.get_artifact_content(artifact['key'])
        #             try:
        #                 parsed_content = self.parse_artifact_content(artifact['key'], content)
        #                 if isinstance(parsed_content, dict) or isinstance(parsed_content, list):
        #                     st.json(parsed_content)
        #                 else:
        #                     st.code(content)
        #             except:
        #                 st.code(content)
    
    def _render_documentation_artifacts(self, artifacts: List[Dict], migration_id: str) -> None:
        """
        Render Documentation artifacts with tabbed interface.
        
        Args:
            artifacts: List of artifacts
            migration_id: Migration ID
        """
        if not artifacts:
            st.info("No Documentation artifacts found yet.")
            return
        
        # Look for documentation.md file
        documentation_md = None
        for artifact in artifacts:
            if artifact['name'] == 'documentation.md':
                documentation_md = self.get_artifact_content(artifact['key'])
                break
        
        if documentation_md:
            # Parse documentation into sections
            doc_sections = self._parse_documentation(documentation_md)
            
            # Create tabs for documentation sections
            tab1, tab2 = st.tabs([
                "Overview", 
                "Migration Steps"
            ])
            
            # Overview Tab
            with tab1:
                st.markdown("## Service Overview")
                if "overview" in doc_sections:
                    st.markdown(doc_sections["overview"])
                else:
                    st.info("No overview information available.")
                    
                if "architecture" in doc_sections:
                    st.markdown("## Architecture")
                    st.markdown(doc_sections["architecture"])
            
            # Migration Steps Tab
            with tab2:
                st.markdown("## Migration Steps")
                
                if "migration_steps" in doc_sections:
                    st.markdown(doc_sections["migration_steps"])
                else:
                    st.info("No migration steps information available.")
                    
                if "validation" in doc_sections:
                    st.markdown("### Validation Results")
                    st.markdown(doc_sections["validation"])
            
            # Add search functionality in sidebar
            with st.sidebar:
                st.subheader("Documentation Search")
                search_term = st.text_input("Search in documentation")
                if search_term:
                    st.subheader("Search Results")
                    # Search in the documentation
                    import re
                    results = []
                    for section_name, content in doc_sections.items():
                        if search_term.lower() in content.lower():
                            results.append((section_name, content))
                    
                    if results:
                        for section_name, content in results:
                            with st.expander(f"Found in {section_name.replace('_', ' ').title()}"):
                                # Highlight the search term
                                pattern = re.compile(f"({re.escape(search_term)})", re.IGNORECASE)
                                highlighted = pattern.sub(r"**\1**", content)
                                st.markdown(highlighted)
                    else:
                        st.info(f"No results found for '{search_term}'")
        else:
            st.info("Documentation file found but could not be loaded.")
            
        # Display all documentation artifacts
        st.subheader("All Documentation Artifacts")
        for artifact in artifacts:
            with st.expander(artifact['name']):
                content = self.get_artifact_content(artifact['key'])
                if artifact['name'].endswith('.md'):
                    st.markdown(content)
                else:
                    try:
                        parsed_content = self.parse_artifact_content(artifact['key'], content)
                        if isinstance(parsed_content, dict) or isinstance(parsed_content, list):
                            st.json(parsed_content)
                        else:
                            st.code(content)
                    except:
                        st.code(content)
                        
    def _parse_documentation(self, documentation):
        """
        Parse the documentation markdown into logical sections
        
        Args:
            documentation: The full markdown documentation
            
        Returns:
            dict: A dictionary of documentation sections
        """
        sections = {}
        
        # Extract overview (everything before first h2)
        import re
        overview_match = re.search(r'^(.*?)(?=\n## |\Z)', documentation, re.DOTALL)
        if overview_match:
            sections["overview"] = overview_match.group(1).strip()
        
        # Extract architecture section
        arch_match = re.search(r'## Architecture.*?\n(.*?)(?=\n## |\Z)', documentation, re.DOTALL)
        if arch_match:
            sections["architecture"] = arch_match.group(1).strip()
        
        # Extract Dockerfile
        dockerfile_match = re.search(r'```dockerfile\s*(.*?)\s*```', documentation, re.DOTALL)
        if dockerfile_match:
            sections["dockerfile"] = dockerfile_match.group(1).strip()
        
        # Extract container configuration
        container_match = re.search(r'## Container Configuration.*?\n(.*?)(?=\n## |\Z)', documentation, re.DOTALL)
        if container_match:
            sections["container_config"] = container_match.group(1).strip()
        
        # Extract Helm chart information
        helm_match = re.search(r'## (Helm Chart|Kubernetes Resources).*?\n(.*?)(?=\n## |\Z)', documentation, re.DOTALL)
        if helm_match:
            sections["helm_chart"] = helm_match.group(2).strip()
        
        # Extract Chart.yaml
        chart_yaml_match = re.search(r'```yaml\s*# Chart.yaml\s*(.*?)\s*```', documentation, re.DOTALL)
        if chart_yaml_match:
            sections["chart_yaml"] = chart_yaml_match.group(1).strip()
        
        # Extract values.yaml
        values_yaml_match = re.search(r'```yaml\s*# values.yaml\s*(.*?)\s*```', documentation, re.DOTALL)
        if values_yaml_match:
            sections["values_yaml"] = values_yaml_match.group(1).strip()
        
        # Extract Kubernetes resources
        k8s_match = re.search(r'## Kubernetes Resources.*?\n(.*?)(?=\n## |\Z)', documentation, re.DOTALL)
        if k8s_match:
            sections["k8s_resources"] = k8s_match.group(1).strip()
        
        # Extract security information
        security_match = re.search(r'## Security.*?\n(.*?)(?=\n## |\Z)', documentation, re.DOTALL)
        if security_match:
            sections["security"] = security_match.group(1).strip()
        
        # Extract AWS Services Used
        aws_services_match = re.search(r'## AWS Services Used.*?\n(.*?)(?=\n## |\Z)', documentation, re.DOTALL)
        if aws_services_match:
            sections["aws_services"] = aws_services_match.group(1).strip()
        
        # Extract IAM policies
        iam_match = re.search(r'## Generated IAM Policies.*?\n(.*?)(?=\n## |\Z)', documentation, re.DOTALL)
        if iam_match:
            sections["iam_policies"] = iam_match.group(1).strip()
        
        # Extract migration steps
        steps_match = re.search(r'## Migration Steps.*?\n(.*?)(?=\n## |\Z)', documentation, re.DOTALL)
        if steps_match:
            sections["migration_steps"] = steps_match.group(1).strip()
        
        # Extract validation results
        validation_match = re.search(r'## Validation Results.*?\n(.*?)(?=\n## |\Z)', documentation, re.DOTALL)
        if validation_match:
            sections["validation"] = validation_match.group(1).strip()
        
        return sections
        
    def _extract_iam_policies(self, markdown_text):
        """Extract IAM policies from markdown text"""
        policies = []
        # Look for JSON blocks that contain IAM policies
        import re
        import json
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', markdown_text, re.DOTALL)
        
        if not json_blocks:
            # If no json blocks found with triple backticks, try with just indentation
            json_blocks = re.findall(r'(?<=\n)(\s*\{\s*"Version".*?\}\s*)(?=\n)', markdown_text, re.DOTALL)
        
        for i, block in enumerate(json_blocks):
            try:
                # Clean up the block - remove any leading/trailing whitespace
                clean_block = block.strip()
                policy = json.loads(clean_block)
                
                # Try to extract policy name if available
                if isinstance(policy, dict):
                    # Look for policy name in the policy itself
                    policy_name = None
                    if "policy_name" in policy:
                        policy_name = policy["policy_name"]
                    elif "PolicyName" in policy:
                        policy_name = policy["PolicyName"]
                    
                    # If no name found in policy, try to find it in surrounding text
                    if not policy_name:
                        # Look for a heading before this policy block
                        policy_block_start = markdown_text.find(block)
                        if policy_block_start > 0:
                            preceding_text = markdown_text[:policy_block_start]
                            heading_match = re.search(r'###\s*(.*?)\s*\n', preceding_text[::-1], re.DOTALL)
                            if heading_match:
                                policy_name = heading_match.group(1)[::-1].strip()
                    
                    # If still no name, use a generic one
                    if not policy_name:
                        policy_name = f"Policy {i+1}"
                    
                    policy["policy_name"] = policy_name
                    policies.append(policy)
            except Exception as e:
                # Just skip this block if it can't be parsed
                pass
                
        return policies
    
    def _render_gitlab_artifacts(self, artifacts: List[Dict], migration_id: str) -> None:
        """
        Render GitLab artifacts.
        
        Args:
            artifacts: List of artifacts
            migration_id: Migration ID
        """
        if not artifacts:
            st.info("No GitLab artifacts found yet.")
            return
        
        st.info("GitLab integration will be performed in the final stage.")
        
    def _render_q_assistant(self, migration_id: str, artifacts: Dict) -> None:
        """
        Render Bedrock Assistant interface.
        
        Args:
            migration_id: Migration ID
            artifacts: Dictionary of artifacts by agent type
        """
        st.header("Bedrock Assistant")
        
        # Initialize session state for context and question if not exists
        if 'previous_context' not in st.session_state:
            st.session_state.previous_context = None
        if 'question' not in st.session_state:
            st.session_state.question = ""
        
        # Context selector
        context_type = st.selectbox(
            "Select Context",
            ["Helm Configuration", "Specific Template", "Canary Deployment"],
            key="context_selector"
        )
        
        # Template selector for specific template context
        selected_template = None
        if context_type == "Specific Template":
            # Get available templates
            template_options = self._get_template_options(artifacts)
            if template_options:
                selected_template = st.selectbox(
                    "Select Template:",
                    options=template_options,
                    key="template_selector"
                )
            else:
                st.warning("No templates found for this migration.")
                return
        
        # Check if context has changed and reset question if it has
        if st.session_state.previous_context != context_type:
            st.session_state.question = ""
            st.session_state.previous_context = context_type
        
        # Get context based on selection
        context = self._get_context_for_q(context_type, migration_id, artifacts, selected_template)
        
        # Show context (collapsible with individual artifacts)
        with st.expander("View Context"):
            if context_type == "Helm Configuration":
                # Parse and display individual artifacts
                self._display_helm_artifacts_context(migration_id, artifacts)
            else:
                st.code(context)
        
        # Question input
        question = st.text_input("Ask Bedrock", value=st.session_state.question, key="q_question")
        st.session_state.question = question
        
        if st.button("Get Recommendation"):
            if not question:
                st.warning("Please enter a question")
                return
            
            # Create a placeholder for the spinner and results
            spinner_placeholder = st.empty()
            
            with spinner_placeholder.container():
                with st.spinner("Getting recommendation from Bedrock..."):
                    # Get Q recommendation
                    recommendation = self._get_q_recommendation(context, question)
            
            # Display recommendation outside the spinner context
            st.markdown(recommendation)
                
    def _get_context_for_q(self, context_type: str, migration_id: str, artifacts: Dict, selected_template: str = None) -> str:
        """
        Get relevant context based on selection.
        
        Args:
            context_type: Type of context to retrieve
            migration_id: Migration ID
            artifacts: Dictionary of artifacts by agent type
            
        Returns:
            str: Context for Amazon Q
        """
        try:
            if not migration_id or not artifacts:
                return "No migration selected or no artifacts available."
            
            if context_type == "Specific Template":
                # Get content for specific template
                if selected_template:
                    for artifact_type, artifact_list in artifacts.items():
                        if artifact_type == 'container_config':
                            for artifact in artifact_list:
                                if artifact['name'] == selected_template:
                                    return f"--- {artifact['name']} ---\n" + self.get_artifact_content(artifact['key'])
                    return f"Template '{selected_template}' not found."
                else:
                    return "No template selected."
                
            elif context_type == "Helm Configuration":
                # Get all Helm chart artifacts including templates, values, and charts
                helm_context = ""
                for artifact_type, artifact_list in artifacts.items():
                    if artifact_type == 'container_config':
                        for artifact in artifact_list:
                            # Include all Helm-related files
                            if any(pattern in artifact['name'].lower() for pattern in [
                                'chart.yaml', 'values', '.yaml', '.yml', 'helmignore', '_helpers.tpl'
                            ]):
                                helm_context += f"\n--- {artifact['name']} ---\n"
                                content = self.get_artifact_content(artifact['key'])
                                helm_context += content + "\n"
                
                return helm_context if helm_context else "No Helm configuration found."
                
            elif context_type == "Canary Deployment":
                # For canary deployment, use only service info + KB patterns (no existing artifacts)
                canary_context = ""
                
                # Get basic service info from DynamoDB
                try:
                    response = self.migration_table.get_item(Key={'migration_id': migration_id})
                    if 'Item' in response:
                        migration = response['Item']
                        if 'service_info' in migration:
                            service_info = migration['service_info']
                            # Extract only essential info for canary generation
                            essential_info = {
                                'service_name': service_info.get('service_name', 'unknown'),
                                'description': service_info.get('description', ''),
                                'ingress': service_info.get('ingress', False),
                                'hostname': service_info.get('hostname', ''),
                                'healthcheck_path': service_info.get('healthcheck_path', '/health'),
                                'service_monitor': service_info.get('service_monitor', False)
                            }
                            canary_context = f"Service Info: {json.dumps(essential_info, indent=2)}\n\n"
                except Exception as e:
                    canary_context = "Service: legacy-transform-api\nHealthcheck: /health\nIngress: enabled\n\n"
                
                canary_context += "Generate canary deployment templates using organizational standards from Knowledge Base."
                return canary_context
            
            else:
                return "Please select a context type."
                
        except Exception as e:
            st.error(f"Error getting context: {str(e)}")
            return f"Error: {str(e)}"
    
    def _display_helm_artifacts_context(self, migration_id: str, artifacts: Dict) -> None:
        """
        Display Helm artifacts in individual sections.
        
        Args:
            migration_id: Migration ID
            artifacts: Dictionary of artifacts by agent type
        """
        try:
            helm_artifacts = []
            for artifact_type, artifact_list in artifacts.items():
                if artifact_type == 'container_config':
                    for artifact in artifact_list:
                        if any(pattern in artifact['name'].lower() for pattern in [
                            'chart.yaml', 'values', '.yaml', '.yml', 'helmignore', '_helpers.tpl'
                        ]):
                            helm_artifacts.append(artifact)
            
            if not helm_artifacts:
                st.info("No Helm artifacts found.")
                return
            
            # Group artifacts by type for better organization
            artifact_groups = {
                'Charts': [],
                'Values': [],
                'Templates': [],
                'Other': []
            }
            
            for artifact in helm_artifacts:
                name = artifact['name'].lower()
                if 'chart.yaml' in name:
                    artifact_groups['Charts'].append(artifact)
                elif 'values' in name:
                    artifact_groups['Values'].append(artifact)
                elif any(t in name for t in ['deployment', 'service', 'ingress', 'configmap', 'secret', '_helpers', 'notes']):
                    artifact_groups['Templates'].append(artifact)
                else:
                    artifact_groups['Other'].append(artifact)
            
            # Display each group
            for group_name, group_artifacts in artifact_groups.items():
                if group_artifacts:
                    st.subheader(f"{group_name} ({len(group_artifacts)} files)")
                    
                    # Create selectbox for artifact selection within each group
                    selected_artifact = st.selectbox(
                        f"Select {group_name} file:",
                        options=[artifact['name'] for artifact in group_artifacts],
                        key=f"artifact_select_{group_name}"
                    )
                    
                    # Display selected artifact content
                    if selected_artifact:
                        selected_artifact_obj = next(a for a in group_artifacts if a['name'] == selected_artifact)
                        content = self.get_artifact_content(selected_artifact_obj['key'])
                        st.code(content, language='yaml')
                            
        except Exception as e:
            st.error(f"Error displaying Helm artifacts: {str(e)}")
    
    def _get_template_options(self, artifacts: Dict) -> List[str]:
        """
        Get list of available template files.
        
        Args:
            artifacts: Dictionary of artifacts by agent type
            
        Returns:
            List of template file names
        """
        template_options = []
        for artifact_type, artifact_list in artifacts.items():
            if artifact_type == 'container_config':
                for artifact in artifact_list:
                    if any(pattern in artifact['name'].lower() for pattern in [
                        'chart.yaml', 'values', '.yaml', '.yml', 'helmignore', '_helpers.tpl'
                    ]):
                        template_options.append(artifact['name'])
        return sorted(template_options)
            
    def _get_q_recommendation(self, context: str, question: str) -> str:
        """
        Get recommendations from Amazon Q via Bedrock with Knowledge Base integration.
        
        Args:
            context: Context for the question
            question: User's question
            
        Returns:
            str: Amazon Q's recommendation
        """
        try:
            # Create focused query for knowledge base retrieval
            kb_query = f"canary deployment flagger istio {question[:50]}"
            
            # Try to retrieve from Knowledge Base first
            retrieved_content = ""
            try:
                retrieve_response = self.bedrock_agent.retrieve(
                    knowledgeBaseId=self.kb_id,
                    retrievalQuery={
                        'text': kb_query
                    },
                    retrievalConfiguration={
                        'vectorSearchConfiguration': {
                            'numberOfResults': 3
                        }
                    }
                )
                
                # Extract retrieved content
                for result in retrieve_response.get('retrievalResults', []):
                    retrieved_content += f"\n{result['content']['text']}\n"
                    
            except Exception as kb_error:
                logger.warning(f"Knowledge Base retrieval failed: {str(kb_error)}")
                # Continue without KB content
            
            # Combine context with knowledge base content
            if retrieved_content:
                full_context = f"""
                {context}
                
                Organizational Standards and Patterns:
                {retrieved_content}
                """
            else:
                full_context = context
            
            # Create prompt with enhanced context
            prompt = f"""
            Context: {full_context}
            
            Question: {question}
            
            Please provide a detailed answer that incorporates both the migration context and organizational standards. 
            If generating templates or configurations, ensure they follow the established patterns.
            If the context doesn't contain enough information, provide general best practices.
            """

            # Prepare the request
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 25000,
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 50,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        
            # Invoke the model
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            generated_text = response_body.get('content', [{}])[0].get('text', '')
            
            return generated_text.strip()
        
        except Exception as e:
            st.error(f"Error getting Q recommendation: {str(e)}")
            return f"Error: {str(e)}"
    
    def restart_migration(self, migration_id: str) -> bool:
        """
        Restart a failed migration by invoking the supervisor Lambda.
        
        Args:
            migration_id: Migration ID to restart
            
        Returns:
            bool: True if restart was successful, False otherwise
        """
        try:
            # Create payload for Lambda invocation
            payload = {
                "action": "restart_migration",
                "migration_id": migration_id
            }
            
            # Invoke the supervisor Lambda
            response = self.lambda_client.invoke(
                FunctionName=self.supervisor_lambda,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload, cls=DecimalEncoder)
            )
            
            # Check response
            status_code = response['StatusCode']
            if status_code >= 200 and status_code < 300:
                logger.info(f"Successfully restarted migration {migration_id}")
                return True
            else:
                logger.error(f"Error invoking Lambda: {status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error restarting migration {migration_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
    def process_pending_migrations(self) -> bool:
        """
        Trigger the supervisor Lambda to process pending migrations.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create payload for Lambda invocation
            payload = {
                "source": "aws.events"  # Use the same format as the CloudWatch Events trigger
            }
            
            # Invoke the supervisor Lambda
            response = self.lambda_client.invoke(
                FunctionName=self.supervisor_lambda,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            # Check response
            status_code = response['StatusCode']
            if status_code >= 200 and status_code < 300:
                logger.info("Successfully triggered processing of pending migrations")
                return True
            else:
                logger.error(f"Error invoking Lambda: {status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing pending migrations: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def invoke_supervisor_lambda(self) -> bool:
        """
        Invoke the supervisor lambda to start a new migration workflow.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Invoke the supervisor lambda with the expected payload format
            response = self.lambda_client.invoke(
                FunctionName=self.supervisor_lambda,
                InvocationType='Event',  # Asynchronous invocation
                Payload=json.dumps({
                    "source": "aws.events",  # Match the format expected by the supervisor
                    "detail-type": "Scheduled Event",
                    "time": datetime.now().isoformat()
                })
            )
            
            # Check if the invocation was successful
            status_code = response.get('StatusCode')
            if status_code == 202:  # 202 Accepted for async invocation
                return True
            else:
                st.error(f"Lambda invocation failed with status code: {status_code}")
                return False
                
        except Exception as e:
            st.error(f"Error invoking supervisor lambda: {str(e)}")
            return False
            return False
    
    def run(self) -> None:
        """Run the dashboard application."""
        # Page config is now set at the top of the file
        
        st.title("Container Migration from EC2 to EKS")
        
        # CRITICAL: Add the migration buttons at the very top, before anything else
        # This ensures they are always visible regardless of any other code execution
        st.markdown("### Start a New Migration")
        
        # Make the button very prominent
        if st.button("🚀 START NEW MIGRATION", key="start_migration_button", 
                    use_container_width=True, type="primary"):
            with st.spinner("Starting migration workflow..."):
                success = self.invoke_supervisor_lambda()
                if success:
                    st.success("Migration workflow started successfully! Refresh in a few moments to see progress.")
                else:
                    st.error("Failed to start migration workflow. Check logs for details.")
        
        # Add refresh button
        if st.button("🔄 Refresh Dashboard", key="refresh_button", use_container_width=True):
            st.rerun()
            
        # Add a separator
        st.markdown("---")
        
        # Load migration data
        df = self.load_migration_data()
        
        if df.empty:
            st.info("No migration data found. Click 'START NEW MIGRATION' to begin the migration process.")
        else:
            # Show migration summary
            self.render_migration_summary(df)
            
            # Show migration details
            self.render_migration_details(df)

def render_feedback_page():
    st.title("Migration Feedback System")
    
    # Initialize feedback component
    feedback_component = FeedbackComponent(region_name=st.session_state.get('region_name', 'us-east-1'))
    
    # Create tabs for different feedback functions
    tab1, tab2 = st.tabs(["Submit Feedback", "View Feedback History"])
    
    with tab1:
        st.header("Submit Feedback on Migration Artifacts")
        
        # Get list of migrations
        migrations = get_migrations_list()
        
        if not migrations:
            st.warning("No migrations found. Please create a migration first.")
            return
        
        # Sort migrations by service name
        sorted_migrations = sorted(migrations, key=lambda m: m.get('service_name', 'Unknown').lower())
        
        # Select migration
        selected_migration = st.selectbox(
            "Select Service", 
            options=[m.get('migration_id') for m in sorted_migrations],
            format_func=lambda x: next((m.get('service_name', 'Unknown') for m in migrations if m.get('migration_id') == x), 'Unknown'),
            key="feedback_migration_selector"  # Add a unique key
        )
        
        if selected_migration:
            # Get migration details
            migration_details = get_migration_details(selected_migration)
            
            if migration_details:
                # Display migration info
                st.subheader(f"Service: {migration_details.get('service_name', 'Unknown')}")
                
                # Get artifacts for this migration
                artifacts = get_migration_artifacts(selected_migration)
                
                if artifacts:
                    # Filter to only show container_config artifacts
                    container_config_artifacts = []
                    for artifact in artifacts:
                        if artifact.get('artifact_type', '').lower() == 'container_config':
                            container_config_artifacts.append(artifact)
                    
                    # Also check for any base/ artifacts regardless of type
                    base_artifacts = []
                    for artifact in artifacts:
                        name = artifact.get('artifact_name', '')
                        if name.startswith('base/') and not name.startswith('base/charts/'):
                            base_artifacts.append(artifact)
                    
                    if container_config_artifacts or base_artifacts:
                        # Use all artifacts (container_config + base artifacts), remove duplicates
                        seen_keys = set()
                        all_artifacts = []
                        for artifact in container_config_artifacts + base_artifacts:
                            s3_key = artifact.get('s3_key', '')
                            if s3_key not in seen_keys:
                                seen_keys.add(s3_key)
                                all_artifacts.append(artifact)
                        
                        # Separate single-app and multi-app artifacts
                        single_app_artifacts = []
                        multi_app_artifacts = {}
                        umbrella_artifacts = []
                        
                        for artifact in all_artifacts:
                            name = artifact.get('artifact_name', '')
                            if name.startswith('base/charts/'):
                                # Multi-app subchart: base/charts/app_name/...
                                parts = name.split('/')
                                if len(parts) >= 3:
                                    app_name = parts[2]
                                    if app_name not in multi_app_artifacts:
                                        multi_app_artifacts[app_name] = []
                                    multi_app_artifacts[app_name].append({
                                        **artifact,
                                        'artifact_name': '/'.join(parts[3:]) if len(parts) > 3 else 'Chart.yaml'
                                    })
                            elif name.startswith('base/'):
                                # Umbrella chart files: base/Chart.yaml, base/values.yaml
                                umbrella_artifacts.append({
                                    **artifact,
                                    'artifact_name': name[5:]  # Remove 'base/' prefix
                                })
                            else:
                                # Single app artifacts
                                single_app_artifacts.append(artifact)
                        
                        # Force umbrella chart display - we know base/ files exist
                        if multi_app_artifacts or umbrella_artifacts or any(art.get('artifact_name', '').startswith('base/') for art in all_artifacts):
                            # Multi-app structure
                            tab_names = ["Umbrella Chart"] + list(multi_app_artifacts.keys())
                            tabs = st.tabs(tab_names)
                            
                            # Umbrella Chart tab
                            with tabs[0]:
                                st.subheader("Umbrella Chart (Multi-App)")
                                if umbrella_artifacts:
                                    for artifact in umbrella_artifacts:
                                        with st.expander(f"📄 {artifact['artifact_name']}"):
                                            content = get_artifact_content(artifact['s3_key'])
                                            if artifact['artifact_name'].endswith('.yaml') or artifact['artifact_name'].endswith('.yml'):
                                                st.code(content, language="yaml")
                                            else:
                                                st.code(content)
                                else:
                                    st.info("No umbrella chart files found.")
                            
                            # Individual app tabs
                            for i, (app_name, app_artifacts) in enumerate(multi_app_artifacts.items(), 1):
                                with tabs[i]:
                                    st.subheader(f"Application: {app_name}")
                                    for artifact in app_artifacts:
                                        with st.expander(f"📄 {artifact['artifact_name']}"):
                                            content = get_artifact_content(artifact['s3_key'])
                                            if artifact['artifact_name'].endswith('.yaml') or artifact['artifact_name'].endswith('.yml'):
                                                st.code(content, language="yaml")
                                            else:
                                                st.code(content)
                        else:
                            # Single app structure - use original layout
                            # Create expandable section for container config artifacts
                            with st.expander(f"Container Configuration ({len(container_config_artifacts)})", expanded=True):
                                # Sort artifacts to ensure Dockerfile is first, followed by Chart files
                                sorted_artifacts = sorted(container_config_artifacts, 
                                                     key=lambda x: (
                                                         0 if 'dockerfile' in x.get('artifact_name', '').lower() else
                                                         1 if 'chart' in x.get('artifact_name', '').lower() else 2,
                                                         x.get('artifact_name', '')
                                                     ))
                            
                            # Create tabs for artifacts
                            artifact_tabs = st.tabs([artifact.get('artifact_name', 'Unnamed Artifact') for artifact in sorted_artifacts])
                            
                            # Initialize feedback state variables if not already set
                            if 'feedback_state' not in st.session_state:
                                st.session_state.feedback_state = {
                                    'show_feedback': False,
                                    'selected_artifact': None,
                                    'selected_tab_index': 0
                                }
                            
                            # Populate each tab with artifact content and feedback option
                            for i, (tab, artifact) in enumerate(zip(artifact_tabs, sorted_artifacts)):
                                with tab:
                                    # Check if this is the active tab
                                    is_active_tab = i == st.session_state.feedback_state['selected_tab_index']
                                    
                                    col1, col2 = st.columns([4, 1])
                                    
                                    with col1:
                                        # Remove the redundant artifact name subheader
                                        pass  # Using pass to maintain proper indentation
                                    
                                    with col2:
                                        # Button to provide feedback on this artifact
                                        if st.button(f"Provide Feedback", key=f"feedback_btn_{i}_{artifact.get('artifact_id')}"):
                                            st.session_state.feedback_state = {
                                                'show_feedback': True,
                                                'selected_artifact': artifact,
                                                'selected_tab_index': i
                                            }
                                            st.rerun()  # Force a rerun to update the UI
                                    
                                    # Get and display artifact content
                                    try:
                                        s3_client = boto3.client('s3', region_name=st.session_state.get('region_name', 'us-east-1'))
                                        bucket_name = st.session_state.get('s3_bucket_name', 'eks-migration-zetaglobal')
                                        
                                        response = s3_client.get_object(
                                            Bucket=bucket_name,
                                            Key=artifact.get('s3_key')
                                        )
                                        
                                        content = response['Body'].read().decode('utf-8')
                                        
                                        # Display content based on file type without the "Content:" header
                                        if artifact.get('artifact_name', '').endswith('.yaml') or artifact.get('artifact_name', '').endswith('.yml'):
                                            st.code(content, language="yaml")
                                        elif artifact.get('artifact_name', '').endswith('.json'):
                                            st.code(content, language="json")
                                        elif 'dockerfile' in artifact.get('artifact_name', '').lower():
                                            st.code(content, language="dockerfile")
                                        else:
                                            st.code(content)
                                            
                                    except Exception as e:
                                        st.error(f"Error retrieving artifact content: {str(e)}")
                                    
                                    # Show feedback form if this is the active tab and feedback is enabled
                                    if is_active_tab and st.session_state.feedback_state['show_feedback']:
                                        st.markdown("---")
                                        feedback_component.render_feedback_form(
                                            migration_id=selected_migration,
                                            artifact_type=artifact.get('artifact_type'),
                                            artifact_id=artifact.get('artifact_id'),
                                            artifact_name=artifact.get('artifact_name'),
                                            s3_key=artifact.get('s3_key')
                                        )
                            
                            # Remove the old feedback form display code since we now show it in each tab
                            # if 'selected_artifact' in st.session_state and 'current_tab_index' in st.session_state:
                            #     # Debug info
                            #     st.write(f"Debug - Current tab: {st.session_state.current_tab_index}, Selected tab: {st.session_state.selected_tab_index}")
                            #     
                            #     # Only show feedback form if we're on the same tab as the selected artifact
                            #     if st.session_state.current_tab_index == st.session_state.selected_tab_index:
                            #         artifact = st.session_state.selected_artifact
                            #         st.markdown("---")
                            #         feedback_component.render_feedback_form(
                            #             migration_id=selected_migration,
                            #             artifact_type=artifact.get('artifact_type'),
                            #             artifact_id=artifact.get('artifact_id'),
                            #             artifact_name=artifact.get('artifact_name'),
                            #             s3_key=artifact.get('s3_key')
                            #         )
                    else:
                        st.info("No container configuration artifacts found for this service.")
                else:
                    st.info("No artifacts found for this service.")
    
    with tab2:
        # Remove the redundant header
        # st.header("Feedback History")
        
        # Get list of migrations
        migrations = get_migrations_list()
        
        if migrations:
            # Sort migrations by service name
            sorted_migrations = sorted(migrations, key=lambda m: m.get('service_name', 'Unknown').lower())
            
            # Create options list with ALL as the first option
            migration_options = ['ALL'] + [m.get('migration_id') for m in sorted_migrations]
            
            # Create a format function that handles the ALL option
            def format_migration_option(option):
                if option == 'ALL':
                    return "ALL SERVICES"
                return next((m.get('service_name', 'Unknown') for m in migrations if m.get('migration_id') == option), 'Unknown')
            
            # Select migration with ALL as default
            selected_migration = st.selectbox(
                "Select Service", 
                options=migration_options,
                format_func=format_migration_option,
                key="history_migration_select",
                index=0  # Set default to first option (ALL)
            )
            
            # Pass the selected migration (or ALL) to the feedback component
            feedback_component.render_feedback_history(migration_id=selected_migration)
        else:
            st.warning("No services found.")

# Helper function to get list of migrations
def get_migrations_list():
    try:
        dynamodb = boto3.resource('dynamodb', region_name=st.session_state.get('region_name', 'us-east-1'))
        table = dynamodb.Table(st.session_state.get('dynamodb_table_name', 'eks-migration-zetaglobal'))
        
        response = table.scan()
        return response.get('Items', [])
    except Exception as e:
        st.error(f"Error fetching migrations: {str(e)}")
        return []

# Helper function to get migration details
def get_migration_details(migration_id):
    try:
        dynamodb = boto3.resource('dynamodb', region_name=st.session_state.get('region_name', 'us-east-1'))
        table = dynamodb.Table(st.session_state.get('dynamodb_table_name', 'eks-migration-zetaglobal'))
        
        response = table.get_item(Key={'migration_id': migration_id})
        return response.get('Item')
    except Exception as e:
        st.error(f"Error fetching migration details: {str(e)}")
        return None

# Helper function to get artifacts for a migration
def get_migration_artifacts(migration_id):
    try:
        # Get S3 client with the correct region
        s3_client = boto3.client('s3', region_name=st.session_state.get('region_name', 'us-east-1'))
        bucket_name = st.session_state.get('s3_bucket_name', 'eks-migration-zetaglobal')
        
        try:
            # Check if bucket exists first
            s3_client.head_bucket(Bucket=bucket_name)
        except Exception as bucket_error:
            st.error(f"S3 bucket '{bucket_name}' does not exist or you don't have access to it: {str(bucket_error)}")
            st.info("Make sure the bucket name is correct in the session state or environment variables.")
            return []
        
        # List objects with the migration_id prefix under artifacts directory
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=f"artifacts/{migration_id}/"
        )
        
        artifacts = []
        # Check if Contents exists in the response
        if 'Contents' in response:
            for obj in response['Contents']:
                # Parse the key to extract artifact info
                key_parts = obj['Key'].split('/')
                if len(key_parts) >= 4:  # artifacts/migration_id/artifact_type/filename_or_path
                    artifact_type = key_parts[2]
                    # Handle nested paths like base/Chart.yaml or base/charts/app/file.yaml
                    artifact_path = '/'.join(key_parts[3:])
                    
                    # Include all files, not just YAML
                    artifact_id = artifact_path.replace('/', '_').split('.')[0]
                    
                    artifacts.append({
                        'artifact_id': artifact_id,
                        'artifact_type': artifact_type,
                        'artifact_name': artifact_path,  # Full path including base/ prefix
                        's3_key': obj['Key']
                    })
        
        return artifacts
    except Exception as e:
        st.error(f"Error fetching artifacts: {str(e)}")
        return []
        
        return artifacts
    except Exception as e:
        st.error(f"Error fetching artifacts: {str(e)}")
        return []

# Define home page rendering function
def render_home_page():
    st.title("Container Migration Dashboard")
    
    st.markdown("""
    ## Welcome to the Container Migration Dashboard
    
    This dashboard helps you monitor and manage the migration of containerized applications from EC2 to Amazon EKS.
    
    ### Key Features:
    
    - **Migration Tracking**: Monitor the status and progress of ongoing migrations in real-time
    - **Artifact Management**: View, compare, and download generated Dockerfiles, Helm charts, and Kubernetes manifests
    - **Security Analysis**: Review security checks, compliance validations, and vulnerability assessments
    - **Feedback System**: Provide targeted feedback on migration artifacts to improve future migrations
    - **Deployment Validation**: Verify the correctness of Kubernetes configurations before deployment
    - **Historical Tracking**: Access complete history of previous migrations and their outcomes
    """)
    
    # Initialize the dashboard
    dashboard = DashboardApp()
    
    # Load migration data
    df = dashboard.load_migration_data()
    
    # Show migration summary
    st.header("Migration Summary")
    dashboard.render_migration_summary(df)

# Define migrations page rendering function
def render_migrations_page():
    st.title("Migration Management")
    
    # Initialize the dashboard
    dashboard = DashboardApp()
    
    # Add Start New Migration button at the top of the migrations page - MAKE IT VERY PROMINENT
    # st.markdown("### Start a New Migration")
    
    # Use a large, primary button that spans the full width
    if st.button("🚀 START NEW MIGRATION", key="migrations_page_start_btn", 
                use_container_width=True, type="primary"):
        with st.spinner("Starting migration workflow..."):
            success = dashboard.invoke_supervisor_lambda()
            if success:
                st.success("Migration workflow started successfully! Refresh in a few moments to see progress.")
            else:
                st.error("Failed to start migration workflow. Check logs for details.")
    
    # Add refresh button below
    # if st.button("🔄 Refresh Data", key="migrations_page_refresh_btn", use_container_width=True):
    #     st.rerun()
    
    # Add a separator
    st.markdown("---")
    
    # Load migration data
    df = dashboard.load_migration_data()
    
    # Show migration details
    dashboard.render_migration_details(df)

# Settings page has been removed to prevent misconfiguration

# Define the sidebar navigation function
def render_sidebar():
    # Get the current page from session state or default to 'Home'
    current_page = st.session_state.get('current_page', 'Home')
    
    # Define all available pages
    pages = {
        'Home': render_home_page,
        'Migrations': render_migrations_page,
        'Feedback': render_feedback_page
    }
    
    # Create the sidebar navigation
    st.sidebar.title('Navigation')
    
    # Use a fixed key for the radio button
    selection = st.sidebar.radio(
        'Go to', 
        list(pages.keys()), 
        key="sidebar_navigation",
        index=list(pages.keys()).index(current_page) if current_page in pages else 0
    )
    
    # Update the current page in session state
    if selection != current_page:
        st.session_state.current_page = selection
        st.rerun()
    
    # Call the selected page function
    pages[selection]()

# Main function to initialize the dashboard
def main():
    """Main function to run the dashboard application."""
    # Initialize fixed AWS settings
    if 'region_name' not in st.session_state:
        st.session_state.region_name = 'us-east-1'
    if 'dynamodb_table_name' not in st.session_state:
        st.session_state.dynamodb_table_name = 'eks-migration-zetaglobal'
    if 's3_bucket_name' not in st.session_state:
        st.session_state.s3_bucket_name = 'eks-migration-zetaglobal'
    if 'feedback_table_name' not in st.session_state:
        st.session_state.feedback_table_name = 'eks-migration-zetaglobal-feedback'
    if 'feedback_function_name' not in st.session_state:
        st.session_state.feedback_function_name = 'eks-migration-zetaglobal-feedback'
    
    # Initialize feedback component
    global feedback_component
    feedback_component = FeedbackComponent()
    
    # Use the sidebar navigation
    render_sidebar()

# Run the main function when the script is executed
if __name__ == '__main__':
    main()
