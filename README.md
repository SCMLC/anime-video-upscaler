# Anime Video Upscaler (Real-ESRGAN + GPU Pipeline)

A high-performance Linux/Bash automated tool designed to restore and upscale low-resolution anime videos to **720p** with crisp details. 

By leveraging **Real-ESRGAN (`realesr-animevideov3`)**, this pipeline executes a full **4x AI super-resolution reconstruction** in GPU memory (VRAM) before downsampling directly to 720p via Bicubic interpolation. This prevents feature collapse—such as facial artifacts or distorted line art in distant characters—while optimizing encode size and processing speed.

---

## Key Features

* **One-Step Bash Wrapper**: Automatically manages system dependencies, sets up a dedicated PyTorch virtual environment, and handles the processing workflow.
* **4x AI Reconstruction + GPU Downsampling**: Runs a single 4x pass through the model to restore missing details, then downsamples in VRAM to 720p (`yuv420p`).
* **Direct GPU Processing**: Pure PyTorch/CUDA tensor operations avoid unnecessary CPU-GPU memory roundtrips.
* **Optimized NVENC Encoding**: Configured with Hardware H.264 (`h264_nvenc`), Constrained Quality (`-cq 26`), and rate limits (`-maxrate 2.2M`) to achieve visually lossy-free output in lightweight file sizes (~250–350 MB per 20-min episode).
* **High Compatibility**: Converts output strictly to standard SDR `yuv420p` color space to prevent overexposure/HDR color shifts and guarantee 100% playback support across TV/mobile players.
* **Batch Frame Processing**: Batches multiple frames directly to GPU for maximum throughput.
* **Resumable Progress**: Automatically logs processed files to `processed_files.txt` to safely resume interrupted runs or process batch folders.

---

## Requirements

### Prerequisites

* **NVIDIA GPU** with CUDA drivers installed (PyTorch utilizes CUDA 12.1).
* **Linux Environment** (Ubuntu/Debian recommended).
* **Sudo Privileges** (required only on the first run to auto-install `ffmpeg`, `git`, and `python3-venv` if missing).

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SCMLC/anime-video-upscaler.git
cd anime-video-upscaler
```

### 2. Run the Upscaler

Pass the target directory containing your anime video files to the script:

```bash
./run_upscale.sh /path/to/your/video_folder
```

> **Note**: The initial run will automatically set up an isolated Python virtual environment (`./venv_video_ai`), install PyTorch with CUDA support, and download the `realesr-animevideov3` model weights.

---

## Workflow Architecture

```text
[ User Action ]
       │
       ▼
[ run_upscale.sh ] ──► 1. Check/Install System Dependencies (ffmpeg, git)
       │           ──► 2. Setup/Activate Python Virtual Env (venv_video_ai)
       │           ──► 3. Install PyTorch (CUDA 12.1) & Real-ESRGAN
       │
       ▼
[ process_videos.py ]
       │
       ├─► [ FFmpeg Ingestion ] ──► Decodes raw BGR24 frame stream
       ├─► [ PyTorch CUDA Tensor ] ──► Batched 4x AI Inference (realesr-animevideov3)
       ├─► [ VRAM Downsampling ] ──► Bicubic interpolation to 720p
       └─► [ FFmpeg NVENC H.264 ] ──► Encodes yuv420p output video
```

---

## Configuration & Tuning

You can modify the constants at the top of `process_videos.py` to match your hardware setup:

```python
BATCH_SIZE = 4       # Increase to 8 if you have 12GB+ VRAM
TARGET_HEIGHT = 720  # Target output resolution height (720p)
```

To tweak the FFmpeg output encoding bitrate/quality, adjust `ffmpeg_out_cmd` inside `process_videos.py`:

* **Higher Quality (Larger File Size)**: Change `-cq 26` to `-cq 23` and `-maxrate 2.2M` to `-maxrate 3M`.
* **Lower File Size**: Change `-cq 26` to `-cq 28` and `-maxrate 2.2M` to `-maxrate 1.8M`.

---

## Supported Formats

Supports common video containers: `.mp4`, `.mkv`, `.avi`, `.mov`, `.flv`, `.wmv`, `.rmvb`, `.webm`, `.ts`.

Outputs are generated in the same directory appended with `[AI_720P]`. Legacy formats (such as `.rmvb`, `.avi`) are automatically remuxed into `.mp4` with audio re-encoded to AAC.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
