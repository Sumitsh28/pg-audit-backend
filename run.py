import uvicorn
import os
import sys

# --- START OF FIX ---
# Get the absolute path of the directory containing this script (the 'backend' folder)
# and add it to the Python path. This ensures that the 'app' module can be found.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
# --- END OF FIX ---

if __name__ == "__main__":
    # Get the port from environment variables, defaulting to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # This command should now work because we've fixed the path.
    # It tells uvicorn to look inside the 'app' module for a 'main.py' file,
    # and inside that file for the FastAPI instance named 'app'.
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)