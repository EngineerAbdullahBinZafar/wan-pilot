"""
Web Workbench UI Server for Wan-Video Pilot.
Serves static studio assets over local HTTP server and optionally launches browser.
"""

import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Optional


class StudioHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        ui_dir = os.path.join(os.path.dirname(__file__), "..", "ui")
        if not os.path.exists(ui_dir):
            ui_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
        super().__init__(*args, directory=os.path.abspath(ui_dir), **kwargs)

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging in test environments
        pass


def launch_ui_server(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True):
    """Launches the Web Workbench local HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, StudioHTTPRequestHandler)
    url = f"http://{host}:{port}/"
    print(f"Wan-Video Pilot Studio Workbench running at {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down studio server...")
        httpd.server_close()


def start_server(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False):
    """Alias for launch_ui_server."""
    launch_ui_server(host=host, port=port, open_browser=open_browser)


if __name__ == "__main__":
    launch_ui_server()
