#!/bin/bash
# Smart Attendance Application run script
# This script launches the application using the verified conda environment python path.

echo "Starting Smart Attendance System Server..."
echo "Using environment interpreter: /opt/anaconda3/envs/antigravity_dl/bin/python"

# Run Flask server
/opt/anaconda3/envs/antigravity_dl/bin/python app.py
