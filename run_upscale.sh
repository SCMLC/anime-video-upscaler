#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Error: Please provide a video directory path!"
    echo "Usage: ./run_upscale.sh /path/to/video_folder"
    exit 1
fi

TARGET_DIR="$1"
VENV_DIR="./venv_video_ai"

# 1. Check and install system dependencies
if ! command -v ffmpeg &> /dev/null || ! command -v git &> /dev/null; then
    echo "Installing system dependencies: ffmpeg, git, python3-venv..."
    sudo apt update && sudo apt install -y ffmpeg git python3-venv
fi

# 2. Create dedicated Python virtual environment (venv)
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Creating dedicated Python virtual environment ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
fi

echo "[2/3] Activating venv and verifying/installing dependencies..."
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
BASICSR_EXT=FALSE pip install git+https://github.com/XPixelGroup/BasicSR.git --no-build-isolation
pip install realesrgan opencv-python tqdm

# 3. Execute Anime AI upscaling
echo "[3/3] Starting Anime AI upscaling to 720p..."
python3 process_videos.py "$TARGET_DIR"

deactivate
echo "Done! Anime video processing completed."
