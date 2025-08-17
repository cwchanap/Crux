"""
Cloudflare Worker entry point for drum transcription API
"""
from js import Response
import json

async def on_fetch(request, env):
    """
    Handle incoming requests to the worker
    This is a lightweight proxy that forwards requests to the main FastAPI app
    """
    url = request.url
    path = url.split(request.headers.get("host"))[1] if request.headers.get("host") else "/"
    
    # Handle static files
    if not path.startswith("/api/"):
        # Serve from the static bucket
        return env.ASSETS.fetch(request)
    
    # For API requests, we need to handle them differently
    # In production, these would be forwarded to a separate service
    # For now, return a message indicating the endpoint
    
    if path == "/api/health":
        return Response.new(
            json.dumps({"status": "healthy", "worker": "python"}),
            headers={"content-type": "application/json"}
        )
    
    # Return 501 Not Implemented for actual transcription endpoints
    # These require heavy processing and should run on a separate service
    return Response.new(
        json.dumps({
            "error": "This endpoint requires heavy processing and should be accessed via the main service",
            "suggestion": "Deploy the FastAPI app on a GPU-enabled service and proxy requests from this worker"
        }),
        status=501,
        headers={"content-type": "application/json"}
    )
