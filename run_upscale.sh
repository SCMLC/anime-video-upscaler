#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "錯誤：請傳入影片資料夾路徑！"
    echo "用法：./run_upscale.sh /path/to/video_folder"
    exit 1
fi

TARGET_DIR="$1"
VENV_DIR="./venv_video_ai"

# 1. 檢查並安裝系統基礎套件
if ! command -v ffmpeg &> /dev/null || ! command -v git &> /dev/null; then
    echo "正在安裝系統依賴 ffmpeg, git, python3-venv..."
    sudo apt update && sudo apt install -y ffmpeg git python3-venv
fi

# 2. 建立獨立虛擬環境 venv
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] 建立獨立 Python 虛擬環境 ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
fi

echo "[2/3] 載入 venv 並檢查/安裝所需套件..."
source "$VENV_DIR/bin/activate"

pip install --upgrade pip 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 
# pip install git+https://github.com/XPixelGroup/BasicSR.git 
BASICSR_EXT=FALSE pip install git+https://github.com/XPixelGroup/BasicSR.git --no-build-isolation
pip install realesrgan opencv-python tqdm

# 3. 執行動漫專用 AI 轉檔
echo "[3/3] 開始執行動漫 AI 畫質提升至 720p..."
python3 process_videos.py "$TARGET_DIR"

deactivate
echo "完成！動漫影片處理完畢。"
