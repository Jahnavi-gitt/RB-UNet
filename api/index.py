"""Vercel Serverless Function entry point for RB-UNet.

Exposes both 'app' (WSGI application) and 'handler' (BaseHTTPRequestHandler)
to ensure seamless routing across any Vercel Python runtime configuration.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path so 'src' and 'app' can be imported reliably on Vercel
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import (
    app,
    application,
    handler,
    RBUNetServerHandler,
    get_status_payload,
    get_results_payload,
    segment_image_payload,
    live_robustness_payload,
)

__all__ = [
    "app",
    "application",
    "handler",
    "RBUNetServerHandler",
    "get_status_payload",
    "get_results_payload",
    "segment_image_payload",
    "live_robustness_payload",
]
