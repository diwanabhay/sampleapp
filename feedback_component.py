import streamlit as st
import boto3
import json
import uuid
from datetime import datetime
import pandas as pd
import traceback

class FeedbackComponent:
    def __init__(self, region_name="us-east-1"):
        self.region_name = region_name
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.lambda_client = boto3.client('lambda', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        
        # Get table name from environment or use default
        self.feedback_table_name = st.session_state.get('feedback_table_name', 'container-migration-ec2eks-feedback')
        self.artifacts_bucket = st.session_state.get('s3_bucket_name', 'container-migration-ec2eks-sundarsm')
        self.feedback_function_name = st.session_state.get('feedback_function_name', 'container-migration-ec2eks-feedback')
    
    def render_feedback_form(self, migration_id, artifact_type, artifact_id, artifact_name=None, version="v1", s3_key=None):
        """Render a feedback form for a specific artifact"""
        # Common feedback tags based on artifact type
        tag_options = []
        if artifact_type == 'container_config':
            if 'dockerfile' in artifact_id.lower():
                tag_options = ['base_image', 'dependencies', 'build_steps', 'security', 'optimization']
            elif 'chart' in artifact_id.lower() or 'helm' in artifact_id.lower():
                tag_options = ['resources', 'configuration', 'dependencies', 'security']
            else:
                tag_options = ['configuration', 'resources', 'security', 'networking']
        elif artifact_type == 'security':
            tag_options = ['vulnerability', 'compliance', 'best_practice', 'remediation']
        elif artifact_type == 'validation':
            tag_options = ['correctness', 'performance', 'reliability', 'best_practice']
                # artifact_key = f"artifacts/{migration_id}/{artifact_type}/{artifact_id}"
                
                # # Add file extension if artifact_name is available
                # if artifact_name:
                #     # Store the original artifact_id for later use in apply_feedback
                #     if 'artifact_id_mapping' not in st.session_state:
                #         st.session_state.artifact_id_mapping = {}
                    
                #     # Map the artifact_name to the artifact_id for this migration
                #     key = f"{migration_id}_{artifact_name}"
        # Display artifact content if available - but don't use an expander since we might be inside one already
        try:
            # If s3_key is provided, use it directly
            if s3_key:
                artifact_key = s3_key
            else:
                # Otherwise construct the key using the standard pattern
                if artifact_name:
                    # If we have a specific artifact name, use it
                    if 'artifact_id_mapping' not in st.session_state:
                        st.session_state.artifact_id_mapping = {}
                    
                    key = f"{migration_id}_{artifact_type}_{artifact_name}"
                    st.session_state.artifact_id_mapping[key] = artifact_id
                    
                    artifact_key = f"artifacts/{migration_id}/{artifact_type}/{artifact_name}"
            
            # st.info(f"Attempting to load artifact from: {artifact_key}")
            
            artifact_response = self.s3_client.get_object(
                Bucket=self.artifacts_bucket,
                Key=artifact_key
            )
            artifact_content = artifact_response['Body'].read().decode('utf-8')
            
            # Determine language for syntax highlighting
            language = "yaml"
            if artifact_name:
                if artifact_name.endswith('.json'):
                    language = "json"
                elif 'dockerfile' in artifact_name.lower():
                    language = "dockerfile"
                elif artifact_name.endswith('.py'):
                    language = "python"
            
            # st.markdown("### Artifact Content:")
            # st.code(artifact_content, language=language)
            
        except Exception as e:
            st.warning(f"Could not load artifact content: {str(e)}")
            st.info(f"Bucket: {self.artifacts_bucket}, Key: {artifact_key}")
            st.info("Make sure the S3 bucket exists and contains the expected artifacts.")
        
        # Feedback form
        with st.form(key=f"feedback_form_{artifact_id}"):
            rating = st.slider("Rating", min_value=1, max_value=5, value=3, 
                              help="Rate the quality of this artifact (1=Poor, 5=Excellent)")
            
            comments = st.text_area("Feedback and Improvement Suggestions", 
                                   placeholder="Please provide your feedback and specific improvement suggestions for this artifact...",
                                   height=200)
            # Initialize session state for feedback if not already set
            if 'last_feedback' not in st.session_state:
                st.session_state.last_feedback = None
            
            # Tag selection if we have tag options
            selected_tags = []
            if tag_options:
                selected_tags = st.multiselect("Categorize your feedback", options=tag_options)
            
            # Apply to all similar services checkbox
            applies_to_all = st.checkbox("Apply this feedback to all similar services", 
                                        help="Check this if your feedback should be applied to all similar services")
            
            submit_button = st.form_submit_button("Submit Feedback")
            
            if submit_button:
                if not comments:
                    st.error("Please provide feedback before submitting.")
                    return
                
                # Prepare feedback payload
                feedback_payload = {
                    "action": "submit_feedback",
                    "migration_id": migration_id,
                    "artifact_type": artifact_type,
                    "artifact_id": artifact_id,
                    "version": version,
                    "feedback": {
                        "rating": rating,
                        "comments": comments,
                        # Remove suggestions field completely since we're not using it anymore
                        "feedback_tags": selected_tags,
                        "applies_to_all": applies_to_all,
                        "status": "pending"
                    }
                }
                
                try:
                    # Invoke Lambda function to process feedback
                    response = self.lambda_client.invoke(
                        FunctionName=self.feedback_function_name,
                        InvocationType='RequestResponse',
                        Payload=json.dumps(feedback_payload)
                    )
                    
                    result = json.loads(response['Payload'].read().decode())
                    
                    if 'statusCode' in result and result['statusCode'] == 200:
                        body = json.loads(result['body'])
                        st.success(f"Feedback submitted successfully!")
                        
                        # Store the feedback ID and other info for later use
                        st.session_state.last_feedback = {
                            'id': body['feedback_id'],
                            'migration_id': migration_id,
                            'artifact_type': artifact_type,
                            'artifact_id': artifact_id,
                            'artifact_name': artifact_name  # Store the artifact name too
                        }
                    else:
                        st.error(f"Error submitting feedback: {result}")
                except Exception as e:
                    st.error(f"Error processing feedback: {str(e)}")
                
        # Create a container for status messages outside the form
        if 'feedback_status_container' not in st.session_state:
            st.session_state.feedback_status_container = st.empty()
                
        # After form submission, show a button outside the form to apply feedback
        if st.session_state.get('last_feedback'):
            feedback = st.session_state.last_feedback
            if st.button("Apply Feedback to Generate Improved Version", key=f"apply_btn_{feedback['id']}"):
                self.apply_feedback(
                    feedback['migration_id'], 
                    feedback['artifact_type'], 
                    feedback['artifact_id']
                )
    
    def render_feedback_history(self, migration_id=None, artifact_id=None):
        """Display feedback history for a migration or artifact in a table format with delete buttons"""
        if not migration_id and not artifact_id:
            st.warning("Please specify either a migration ID or artifact ID to view feedback history.")
            return
        
        try:
            # Prepare payload for Lambda
            payload = {
                "action": "get_feedback"
            }
            
            if migration_id:
                payload["migration_id"] = migration_id
            if artifact_id:
                payload["artifact_id"] = artifact_id
            
            # Show a message when viewing all feedback
            # if migration_id == 'ALL':
            #     st.info("Showing feedback from all services")
                
            # Add debug info
            if st.session_state.get('debug_mode', False):
                st.write("Debug - Payload sent to Lambda:")
                st.write(payload)
            
            # Invoke Lambda function to get feedback
            response = self.lambda_client.invoke(
                FunctionName=self.feedback_function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            result = json.loads(response['Payload'].read().decode())
            
            # Add debug info
            if st.session_state.get('debug_mode', False):
                st.write("Debug - Lambda response:")
                st.write(result)
            
            if 'statusCode' in result and result['statusCode'] == 200:
                body = json.loads(result['body'])
                feedback_items = body.get('feedback_items', [])
                
                if not feedback_items:
                    st.info("No feedback found.")
                    return
                
            # Add debug info
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug - Found {len(feedback_items)} feedback items")
            
            # Convert to DataFrame for easier display
            df = pd.DataFrame(feedback_items)
            
            # Sort by timestamp (newest first)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False)
            
            # Create a clean display version of the DataFrame
            display_df = df.copy()
            
            # Format columns for display
            if 'timestamp' in display_df.columns:
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            if 'artifact_type' in display_df.columns:
                display_df['artifact_type'] = display_df['artifact_type'].apply(
                    lambda x: x.replace('_', ' ').title() if isinstance(x, str) else x
                )
                
                # Format status column to show proper status
                if 'status' in display_df.columns:
                    # Map status values to display values
                    status_map = {
                        'pending': 'Pending',
                        'implemented': 'Implemented',
                        'applied': 'Implemented'  # Map 'applied' to 'Implemented' as well
                    }
                    display_df['status'] = display_df['status'].str.lower().map(status_map).fillna('Pending')
                
                # Select and rename columns for display
                columns_to_display = ['timestamp', 'artifact_type', 'artifact_id', 'rating', 'comments', 'status']
                column_names = {
                    'timestamp': 'Date & Time',
                    'artifact_type': 'Artifact Type',
                    'artifact_id': 'Component',
                    'rating': 'Rating',
                    'comments': 'Comments',
                    'status': 'Status'
                }
                
                # Filter columns that exist in the DataFrame
                existing_columns = [col for col in columns_to_display if col in display_df.columns]
                display_df = display_df[existing_columns].rename(columns={col: column_names.get(col, col) for col in existing_columns})
                
                # Add service name to display_df if viewing ALL services
                if migration_id == 'ALL' and 'migration_id' in df.columns:
                    # Create a mapping of migration_id to service_name
                    try:
                        # Get DynamoDB client
                        dynamodb = boto3.resource('dynamodb', region_name=self.region_name)
                        migration_table = dynamodb.Table(st.session_state.get('dynamodb_table_name', 'container-migration-ec2eks-sundarsm'))
                        
                        # Create a dictionary to store service names
                        service_names = {}
                        
                        # Get unique migration IDs
                        unique_migration_ids = df['migration_id'].unique()
                        
                        # Fetch service names for each migration ID
                        for mid in unique_migration_ids:
                            try:
                                response = migration_table.get_item(Key={'migration_id': mid})
                                if 'Item' in response and 'service_name' in response['Item']:
                                    service_names[mid] = response['Item']['service_name']
                                else:
                                    service_names[mid] = 'Unknown'
                            except Exception:
                                service_names[mid] = 'Unknown'
                        
                        # Add service name column to display_df
                        display_df['Service'] = df['migration_id'].apply(lambda mid: service_names.get(mid, 'Unknown'))
                        
                        # Reorder columns to put Service near the beginning
                        cols = display_df.columns.tolist()
                        cols.remove('Service')
                        display_df = display_df[['Date & Time', 'Service'] + cols[1:]]
                    except Exception as e:
                        st.warning(f"Could not fetch service names: {str(e)}")
                
                # Add action buttons columns
                display_df['Actions'] = ''
                
                # Display the table header
                st.write("### Feedback History")
                
                # Add CSS for table styling with vertical and horizontal borders
                st.markdown("""
                <style>
                .feedback-table {
                    border: 1px solid #0068C9;
                    width: 100%;
                    margin-bottom: 10px;
                    border-collapse: collapse;
                }
                .feedback-table-row {
                    border-bottom: 1px solid #0068C9;
                }
                .feedback-table-cell {
                    border-right: 1px solid #0068C9;
                    padding: 8px;
                    overflow-wrap: break-word;
                    word-wrap: break-word;
                }
                .feedback-table-header {
                    font-weight: bold;
                    border-bottom: 2px solid #0068C9;
                    border-right: 1px solid #0068C9;
                    padding: 8px;
                    background-color: #f0f2f6;
                }
                .feedback-table-cell:last-child, .feedback-table-header:last-child {
                    border-right: none;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Start table with border
                st.markdown('<div class="feedback-table">', unsafe_allow_html=True)
                
                # Create table headers
                if migration_id == 'ALL' and 'Service' in display_df.columns:
                    cols = st.columns([3, 2, 2, 1, 3, 1, 1, 1])
                    with cols[0]:
                        st.markdown('<div class="feedback-table-header">Date & Time</div>', unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown('<div class="feedback-table-header">Service</div>', unsafe_allow_html=True)
                    with cols[2]:
                        st.markdown('<div class="feedback-table-header">Artifact Type</div>', unsafe_allow_html=True)
                    with cols[3]:
                        st.markdown('<div class="feedback-table-header">Component</div>', unsafe_allow_html=True)
                    with cols[4]:
                        st.markdown('<div class="feedback-table-header">Comments</div>', unsafe_allow_html=True)
                    with cols[5]:
                        st.markdown('<div class="feedback-table-header">Rating</div>', unsafe_allow_html=True)
                    with cols[6]:
                        st.markdown('<div class="feedback-table-header">Status</div>', unsafe_allow_html=True)
                    with cols[7]:
                        st.markdown('<div class="feedback-table-header">Actions</div>', unsafe_allow_html=True)
                else:
                    cols = st.columns([3, 2, 1, 3, 1, 1, 1])
                    with cols[0]:
                        st.markdown('<div class="feedback-table-header">Date & Time</div>', unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown('<div class="feedback-table-header">Artifact Type</div>', unsafe_allow_html=True)
                    with cols[2]:
                        st.markdown('<div class="feedback-table-header">Component</div>', unsafe_allow_html=True)
                    with cols[3]:
                        st.markdown('<div class="feedback-table-header">Comments</div>', unsafe_allow_html=True)
                    with cols[4]:
                        st.markdown('<div class="feedback-table-header">Rating</div>', unsafe_allow_html=True)
                    with cols[5]:
                        st.markdown('<div class="feedback-table-header">Status</div>', unsafe_allow_html=True)
                    with cols[6]:
                        st.markdown('<div class="feedback-table-header">Actions</div>', unsafe_allow_html=True)
                
                # Create table rows
                for i, row in display_df.iterrows():
                    # Get the corresponding original row from df to access all fields
                    orig_row = df.loc[i]
                    
                    # Check if feedback is already implemented
                    is_implemented = row['Status'].lower() in ['implemented', 'applied']
                    
                    # Convert status to emoji with tooltip
                    status_emoji = "✓" if is_implemented else "⏳"
                    status_tooltip = "Implemented" if is_implemented else "Pending"
                    
                    # Create a row with border
                    st.markdown('<div class="feedback-table-row">', unsafe_allow_html=True)
                    
                    if migration_id == 'ALL' and 'Service' in display_df.columns:
                        cols = st.columns([3, 2, 2, 1, 3, 1, 1, 1])
                        
                        with cols[0]:
                            st.markdown(f'<div class="feedback-table-cell">{row["Date & Time"]}</div>', unsafe_allow_html=True)
                        with cols[1]:
                            st.markdown(f'<div class="feedback-table-cell">{row["Service"]}</div>', unsafe_allow_html=True)
                        with cols[2]:
                            st.markdown(f'<div class="feedback-table-cell">{row["Artifact Type"]}</div>', unsafe_allow_html=True)
                        with cols[3]:
                            st.markdown(f'<div class="feedback-table-cell">{row["Component"]}</div>', unsafe_allow_html=True)
                        with cols[4]:
                            st.markdown(f'<div class="feedback-table-cell">{row["Comments"]}</div>', unsafe_allow_html=True)
                        with cols[5]:
                            st.markdown(f'<div class="feedback-table-cell">⭐ {row["Rating"]}</div>', unsafe_allow_html=True)
                        with cols[6]:
                            st.markdown(f'<div class="feedback-table-cell"><span title="{status_tooltip}">{status_emoji}</span></div>', unsafe_allow_html=True)
                        
                        # Action buttons in the last column
                        with cols[7]:
                            st.markdown('<div class="feedback-table-cell">', unsafe_allow_html=True)
                            if is_implemented:
                                # Only show delete button
                                if st.button("🗑️", key=f"delete_{orig_row.get('feedback_id')}", help="Delete this feedback"):
                                    self.delete_feedback(orig_row.get('feedback_id'))
                            else:
                                # Show both apply and delete buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("✅", key=f"apply_{orig_row.get('feedback_id')}", help="Apply this feedback"):
                                        self.apply_feedback(
                                            orig_row.get('migration_id'), 
                                            orig_row.get('artifact_type'), 
                                            orig_row.get('artifact_id')
                                        )
                                with col2:
                                    if st.button("🗑️", key=f"delete_{orig_row.get('feedback_id')}", help="Delete this feedback"):
                                        self.delete_feedback(orig_row.get('feedback_id'))
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        cols = st.columns([3, 2, 1, 3, 1, 1, 1])
                        
                        with cols[0]:
                            st.markdown(f'<div class="feedback-table-cell">{row["Date & Time"]}</div>', unsafe_allow_html=True)
                        with cols[1]:
                            st.markdown(f'<div class="feedback-table-cell">{row["Artifact Type"]}</div>', unsafe_allow_html=True)
                        with cols[2]:
                            st.markdown(f'<div class="feedback-table-cell">{row["Component"]}</div>', unsafe_allow_html=True)
                        with cols[3]:
                            st.markdown(f'<div class="feedback-table-cell">{row["Comments"]}</div>', unsafe_allow_html=True)
                        with cols[4]:
                            st.markdown(f'<div class="feedback-table-cell">⭐ {row["Rating"]}</div>', unsafe_allow_html=True)
                        with cols[5]:
                            st.markdown(f'<div class="feedback-table-cell"><span title="{status_tooltip}">{status_emoji}</span></div>', unsafe_allow_html=True)
                        
                        # Action buttons in the last column
                        with cols[6]:
                            st.markdown('<div class="feedback-table-cell">', unsafe_allow_html=True)
                            if is_implemented:
                                # Only show delete button
                                if st.button("🗑️", key=f"delete_{orig_row.get('feedback_id')}", help="Delete this feedback"):
                                    self.delete_feedback(orig_row.get('feedback_id'))
                            else:
                                # Show both apply and delete buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("✅", key=f"apply_{orig_row.get('feedback_id')}", help="Apply this feedback"):
                                        self.apply_feedback(
                                            orig_row.get('migration_id'), 
                                            orig_row.get('artifact_type'), 
                                            orig_row.get('artifact_id')
                                        )
                                with col2:
                                    if st.button("🗑️", key=f"delete_{orig_row.get('feedback_id')}", help="Delete this feedback"):
                                        self.delete_feedback(orig_row.get('feedback_id'))
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Close the row
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Close the table
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"Error retrieving feedback: {result}")
        except Exception as e:
            st.error(f"Error loading feedback history: {str(e)}")
            st.code(traceback.format_exc())
    
    def delete_feedback(self, feedback_id):
        """Delete a feedback item"""
        try:
            # Prepare payload for Lambda
            payload = {
                "action": "delete_feedback",
                "feedback_id": feedback_id
            }
            
            # Show a spinner while processing
            with st.spinner("Deleting feedback..."):
                # Invoke Lambda function to delete feedback
                response = self.lambda_client.invoke(
                    FunctionName=self.feedback_function_name,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(payload)
                )
                
                result = json.loads(response['Payload'].read().decode())
                
                if 'statusCode' in result and result['statusCode'] == 200:
                    # Display success message
                    st.success("Feedback deleted successfully!")
                    # Force refresh of the page to update the feedback list
                    st.rerun()
                else:
                    # Display error message
                    st.error(f"Error deleting feedback: {result}")
        except Exception as e:
            # Display error message
            st.error(f"Error deleting feedback: {str(e)}")
            st.code(traceback.format_exc())
    
    def apply_feedback(self, migration_id, artifact_type, artifact_id):
        """Apply feedback to generate an improved artifact"""
        try:
            # Create a placeholder for feedback results if it doesn't exist
            if 'feedback_results' not in st.session_state:
                st.session_state.feedback_results = st.empty()
            
            # Show an info message while processing
            with st.spinner("Applying feedback and generating improved artifact..."):
                # Prepare payload for Lambda
                payload = {
                    "action": "apply_feedback",
                    "migration_id": migration_id,
                    "artifact_type": artifact_type,
                    "artifact_id": artifact_id
                }
                
                # Invoke Lambda function to apply feedback
                response = self.lambda_client.invoke(
                    FunctionName=self.feedback_function_name,
                    InvocationType='RequestResponse',
                    Payload=json.dumps(payload)
                )
                
                result = json.loads(response['Payload'].read().decode())
                
                if 'statusCode' in result and result['statusCode'] == 200:
                    body = json.loads(result['body'])
                    
                    # Store the results in session state for display
                    st.session_state.feedback_applied = True
                    st.session_state.feedback_result = body
                    st.session_state.feedback_migration_id = migration_id
                    
                    # Show success message
                    st.success("✅ Feedback applied successfully! New artifact created.")
                    
                    # Display summary of changes
                    st.subheader("Summary of Changes")
                    summary = body.get('summary', 'No summary provided')
                    if summary and summary != "No summary available":
                        st.write(summary)
                    else:
                        st.write("No detailed summary was provided for the changes.")
                    
                    # Display the new artifact
                    new_artifact_key = body.get('new_artifact_key')
                    if new_artifact_key:
                        try:
                            # Get the new artifact content
                            artifact_response = self.s3_client.get_object(
                                Bucket=self.artifacts_bucket,
                                Key=new_artifact_key
                            )
                            new_artifact_content = artifact_response['Body'].read().decode('utf-8')
                    
                            # Display the new artifact
                            st.subheader("Improved Artifact")
                            
                            # Determine language for syntax highlighting
                            language = "yaml"
                            if 'dockerfile' in new_artifact_key.lower():
                                language = "dockerfile"
                            elif new_artifact_key.endswith('.json'):
                                language = "json"
                            
                            st.code(new_artifact_content, language=language)
                            
                            # Force refresh of artifacts in session state
                            if 'artifacts_cache' in st.session_state:
                                # Clear the artifacts cache to force reload
                                if migration_id in st.session_state.artifacts_cache:
                                    del st.session_state.artifacts_cache[migration_id]
                            
                            # Add a button to refresh the page
                            if st.button("Refresh Container Configuration"):
                                st.rerun()
                        except Exception as e:
                            st.warning(f"Could not load new artifact content: {str(e)}")
                else:
                    st.error(f"Error applying feedback: {result}")
        except Exception as e:
            st.error(f"Error applying feedback: {str(e)}")
            st.code(traceback.format_exc())
    
    def render_feedback_dashboard(self):
        """Render a dashboard for feedback analytics"""
        # Remove redundant subheader
        # st.subheader("Feedback Analytics")
        
        try:
            # Scan the feedback table to get all items
            table = self.dynamodb.Table(self.feedback_table_name)
            response = table.scan()
            items = response.get('Items', [])
            
            # Continue scanning if we have more items (pagination)
            while 'LastEvaluatedKey' in response:
                response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
                items.extend(response.get('Items', []))
            
            if not items:
                st.info("No feedback data available for analytics.")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(items)
            
            # Basic analytics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Feedback Count", len(df))
                
                if 'rating' in df.columns and not df['rating'].isna().all():
                    avg_rating = df['rating'].mean()
                    st.metric("Average Rating", f"{avg_rating:.2f}/5.0")
            
            with col2:
                if 'artifact_type' in df.columns and not df['artifact_type'].isna().all():
                    artifact_counts = df['artifact_type'].value_counts()
                    st.write("Feedback by Artifact Type")
                    st.bar_chart(artifact_counts)
            
            # Rating distribution - REMOVED as requested
            # if 'rating' in df.columns:
            #     st.write("Rating Distribution")
            #     rating_counts = df['rating'].value_counts().sort_index()
            #     st.bar_chart(rating_counts)
            
            # Feedback over time
            if 'timestamp' in df.columns and not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date
                
                # Check if we have more than one date to plot
                if len(df['date'].unique()) > 1:
                    feedback_by_date = df.groupby('date').size()
                    
                    st.write("Feedback Submissions Over Time")
                    st.line_chart(feedback_by_date)
                else:
                    # If we only have one date, create a simple bar chart
                    st.write("Feedback Submissions Over Time")
                    date = df['date'].iloc[0]
                    count = len(df)
                    
                    # Create a simple DataFrame with the single date
                    single_date_df = pd.DataFrame({
                        'date': [date],
                        'count': [count]
                    }).set_index('date')
                    
                    st.bar_chart(single_date_df)
            
            # Most common suggestions section removed - will be implemented later
            # st.write("Common Feedback Themes")
            # st.info("This feature will analyze feedback comments to extract common themes and suggestions.")
            
        except Exception as e:
            st.error(f"Error generating feedback analytics: {str(e)}")
    def get_artifact_content(self, key: str) -> str:
        """
        Get the content of an artifact from S3.
        
        Args:
            key: S3 key of the artifact
            
        Returns:
            str: Content of the artifact
        """
        try:
            response = self.s3_client.get_object(Bucket=self.artifacts_bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            return content
        except Exception as e:
            error_msg = f"Could not load artifact content: {str(e)}"
            st.error(error_msg)
            st.info(f"Make sure the S3 bucket '{self.artifacts_bucket}' exists and you have access to it.")
            return error_msg
