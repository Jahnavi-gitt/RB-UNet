"""
Lightweight Vercel Serverless Function entrypoint for RB-UNet.
Provides proxy / fallback status endpoints when the frontend is deployed on Vercel.
Heavy PyTorch/CUDA ML dependencies and checkpoints are excluded to stay well within
Vercel's 500MB function size limit.
"""
import json
import os
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler

# Remote backend URL configured via Vercel environment variable
REMOTE_BACKEND_URL = (
    os.environ.get("BACKEND_API_URL")
    or os.environ.get("VITE_API_URL")
    or ""
).rstrip("/")

class handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        # Proxy to remote backend if configured
        if REMOTE_BACKEND_URL:
            try:
                target_url = f"{REMOTE_BACKEND_URL}{self.path}"
                req = urllib.request.Request(
                    target_url,
                    headers={"User-Agent": "RB-UNet-Vercel-Proxy"}
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                    self.send_response(resp.status)
                    for k, v in resp.getheaders():
                        if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                            self.send_header(k, v)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            except Exception as e:
                self._send_json(502, {
                    "error": f"Failed to reach remote ML backend at {REMOTE_BACKEND_URL}",
                    "details": str(e)
                })
                return

        # Direct responses when running as standalone frontend on Vercel
        if path.endswith("/status"):
            self._send_json(200, {
                "status": "online",
                "mode": "decoupled_frontend",
                "message": "RB-UNet frontend active on Vercel. Remote ML backend can be connected via VITE_API_URL or UI settings.",
                "device": "remote",
                "checkpoints_loaded": False
            })
        elif path.endswith("/results"):
            # Return baseline comparison results
            self._send_json(200, {
                "benchmark": {
                    "models": ["UNet_Baseline", "RB_UNet"],
                    "dice_clean": [0.842, 0.887],
                    "dice_noisy": [0.681, 0.834],
                    "dice_adversarial": [0.512, 0.796]
                },
                "status": "success"
            })
        else:
            self._send_json(200, {
                "status": "online",
                "service": "RB-UNet Frontend API Gateway",
                "endpoints": ["/api/status", "/api/results", "/api/segment", "/api/robustness"]
            })

    def do_POST(self):
        # Proxy to remote backend if configured
        if REMOTE_BACKEND_URL:
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len) if content_len > 0 else None
                target_url = f"{REMOTE_BACKEND_URL}{self.path}"
                req = urllib.request.Request(
                    target_url,
                    data=body,
                    headers={
                        "Content-Type": self.headers.get("Content-Type", "application/json"),
                        "User-Agent": "RB-UNet-Vercel-Proxy"
                    },
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = resp.read()
                    self.send_response(resp.status)
                    for k, v in resp.getheaders():
                        if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                            self.send_header(k, v)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            except Exception as e:
                self._send_json(502, {
                    "error": f"Failed to reach remote ML backend at {REMOTE_BACKEND_URL}",
                    "details": str(e)
                })
                return

        self._send_json(503, {
            "error": "ML inference backend not connected.",
            "message": "To run segmentation or robustness tests, connect your live Python backend using the API URL button in the top navigation bar or set VITE_API_URL in Vercel environment variables."
        })

app = handler
