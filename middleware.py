import streamlit as st
import os
from streamlit.web.server.server import Server
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.state import SessionState
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.runtime import Runtime
from streamlit.web.server import websocket_headers
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppRunnerMiddleware:
    """
    Custom middleware to handle WebSocket connection issues in App Runner.
    This middleware forces Streamlit to use Server-Sent Events (SSE) instead of WebSockets.
    """
    def __init__(self):
        logger.info("Initializing AppRunnerMiddleware")
        
        # Force server-sent events mode
        os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET"] = "false"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        
        # Add custom headers for App Runner
        websocket_headers.WEBSOCKET_HEADERS = {
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Port": "443"
        }
        
        logger.info("AppRunnerMiddleware initialized with environment variables and headers")

# Initialize the middleware
middleware = AppRunnerMiddleware()
logger.info("AppRunnerMiddleware loaded")
