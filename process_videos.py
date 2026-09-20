import os
import sys
import urllib.request
import subprocess
import cv2
import numpy as np
import torch
import torchvision

# Fix PyTorch 2.x compatibility
if not hasattr(torch, 'torchversion'):
    torch.torchversion = torchvision.__version__

# Enable CUDA convolution and memory allocation optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from tqdm import tqdm

# =========================================================================
# 💡 Configuration Section
# =========================================================================
BATCH_SIZE = 4                  # Number of frames fed into GPU per inference pass (reduce to 2 or 1 if VRAM is low)
TARGET_HEIGHT = 720             # Final target height (720p)

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.rmvb', '.webm', '.ts')
LEGACY_EXTENSIONS = ('.rmvb', '.flv', '.wmv', '.avi')
LOG_FILENAME = "processed_files.txt"

# Switch to the deeper RealESRGAN_x4plus_anime_6B model to prevent facial line artifacts
MODEL_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
MODEL_NAME = 'RealESRGAN_x4plus_anime_6B.pth'

def load_processed_log(log_path):
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def append_processed_log(log_path, filename):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{filename}\n")

def download_model_if_needed():
    if not os.path.exists(MODEL_NAME):
        print(f"Downloading anime video AI model ({MODEL_NAME})...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
        print("Download complete!")

def get_video_info(file_path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        file_path
    ]
    try:
        out = subprocess.check_output(cmd).decode().strip().split('\n')
        stream_info = out[0].split(',')
        width = int(stream_info[0])
        height = int(stream_info[1])
        fps_str = stream_info[2]

        if '/' in fps_str:
            num, den = map(float, fps_str.split('/'))
            fps = num / den if den != 0 else 23.976
        else:
            fps = float(fps_str)

        duration = float(out[1]) if len(out) > 1 and out[1] else 0
        total_frames = int(duration * fps) if duration > 0 else 0

        if fps <= 0 or np.isnan(fps) or fps > 120:
            fps = 23.976

        return width, height, fps, total_frames
    except Exception:
        return None, None, 23.976, 0

def process_batch_frames_gpu_direct_downsample(batch_frames, upsampler, target_h):
    """
    Single 4x inference pass + VRAM downsampling to 720p 
    (Prevents facial/feature distortion on distant characters)
    """
    if not batch_frames:
        return []

    # 1. Convert to CUDA Tensor
    imgs = [f[:, :, ::-1] for f in batch_frames]
    imgs_array = np.stack(imgs, axis=0)

    curr_tensor = torch.from_numpy(np.ascontiguousarray(imgs_array.transpose(0, 3, 1, 2))).to(
        upsampler.device, non_blocking=True
    ).float() / 255.0

    if getattr(upsampler, 'half', False):
        curr_tensor = curr_tensor.half()

    model = upsampler.model
    model.eval()

    # Calculate target 720p dimensions (maintaining aspect ratio)
    _, _, orig_h, orig_w = curr_tensor.shape
    scale = target_h / orig_h
    target_w = int(orig_w * scale)
    if target_w % 2 != 0:
        target_w += 1

    # 2. Single 4x inference + Direct GPU Bicubic downsampling (smooths feature linework)
    with torch.no_grad():
        out_4x = model(curr_tensor)

        final_tensor = torch.nn.functional.interpolate(
            out_4x,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        ).clamp_(0, 1)

    # 3. Transfer back to CPU and send to FFmpeg after processing
    final_tensor = final_tensor.data.float().cpu().clamp_(0, 1)
    output_np = final_tensor.numpy().transpose(0, 2, 3, 1)

    outputs = []
    for i in range(len(batch_frames)):
        out_img = output_np[i][:, :, ::-1]
        out_img = (out_img * 255.0).round().astype(np.uint8)
        outputs.append(out_img)

    return outputs

def upscale_video_anime(input_path, output_path, upsampler, width, height, fps, total_frames, force_aac=False):
    scale = TARGET_HEIGHT / height
    new_height = TARGET_HEIGHT
    new_width = int(width * scale)
    if new_width % 2 != 0:
        new_width += 1

    print(f"\n[Anime AI Single 4x (6B Model) -> 720p Downsample Enhancement] {os.path.basename(input_path)}")
    print(f"Resolution change: {height}p -> Downsampled {TARGET_HEIGHT}p (Batch Size = {BATCH_SIZE})")

    audio_codec_args = ['-c:a', 'aac', '-b:a', '192k'] if force_aac else ['-c:a', 'copy']

    ffmpeg_in_cmd = [
        'ffmpeg', '-v', 'error',
        '-i', input_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-vsync', 'cfr',
        '-r', f"{fps:.3f}",
        '-'
    ]

    # 💡 Color space compatibility fix (yuv420p) + File size optimization settings
    ffmpeg_out_cmd = [
        'ffmpeg', '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{new_width}x{new_height}',
        '-pix_fmt', 'bgr24',        # Input: Accept raw BGR data passed from Python
        '-r', f"{fps:.3f}",
        '-i', '-',
        '-i', input_path,
        '-map', '0:v:0',
        '-map', '1:a?',
        '-c:v', 'h264_nvenc',
        '-pix_fmt', 'yuv420p',      # Output: Force standard yuv420p (playback compatibility + smaller file size)
        '-preset', 'p4',
        '-rc', 'vbr',
        '-cq', '26',
        '-maxrate', '2.2M',
        '-bufsize', '4.4M',
    ] + audio_codec_args + [output_path]

    proc_in = subprocess.Popen(ffmpeg_in_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    proc_out = subprocess.Popen(ffmpeg_out_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="AI Processing Progress")
    frame_bytes = width * height * 3
    batch_frames = []

    try:
        while True:
            in_bytes = proc_in.stdout.read(frame_bytes)
            if len(in_bytes) < frame_bytes:
                if batch_frames:
                    out_frames = process_batch_frames_gpu_direct_downsample(batch_frames, upsampler, TARGET_HEIGHT)
                    for out_frame in out_frames:
                        proc_out.stdin.write(out_frame.tobytes())
                    pbar.update(len(batch_frames))
                    batch_frames.clear()
                break

            frame = np.frombuffer(in_bytes, dtype=np.uint8).reshape((height, width, 3))
            batch_frames.append(frame)

            if len(batch_frames) == BATCH_SIZE:
                out_frames = process_batch_frames_gpu_direct_downsample(batch_frames, upsampler, TARGET_HEIGHT)
                for out_frame in out_frames:
                    proc_out.stdin.write(out_frame.tobytes())
                pbar.update(len(batch_frames))
                batch_frames.clear()

            if proc_out.poll() is not None:
                _, stderr_data = proc_out.communicate()
                print(f"\n[FFmpeg Error]:\n{stderr_data.decode('utf-8', errors='ignore')}")
                raise RuntimeError("FFmpeg process terminated unexpectedly")

    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e
    finally:
        if proc_in.stdout:
            proc_in.stdout.close()
        proc_in.wait()
        if proc_out.stdin and not proc_out.stdin.closed:
            proc_out.stdin.close()
        proc_out.wait()
        pbar.close()

def setup_ai_model():
    download_model_if_needed()
    # Initialize RRDBNet for 6B anime architecture
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
        scale=4
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=MODEL_NAME,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0
    )
    return upsampler

def main():
    if len(sys.argv) < 2:
        print("Please provide the target video directory path!")
        sys.exit(1)

    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f"Specified path does not exist: {target_dir}")
        sys.exit(1)

    log_path = os.path.join(target_dir, LOG_FILENAME)
    processed_files = load_processed_log(log_path)

    upsampler = setup_ai_model()

    for fname in sorted(os.listdir(target_dir)):
        if fname == LOG_FILENAME:
            continue

        name, ext = os.path.splitext(fname)
        ext_lower = ext.lower()

        if ext_lower not in VIDEO_EXTENSIONS or name.endswith('[AI_720P]'):
            continue

        if fname in processed_files:
            print(f"[Already processed, skipping] {fname}")
            continue

        input_path = os.path.join(target_dir, fname)

        if ext_lower in LEGACY_EXTENSIONS:
            out_ext = '.mp4'
            force_aac = True
        else:
            out_ext = ext
            force_aac = False

        output_fname = f"{name}[AI_720P]{out_ext}"
        output_path = os.path.join(target_dir, output_fname)

        width, height, fps, total_frames = get_video_info(input_path)
        if height is None:
            continue

        if height < TARGET_HEIGHT:
            try:
                upscale_video_anime(input_path, output_path, upsampler, width, height, fps, total_frames, force_aac=force_aac)
                append_processed_log(log_path, fname)
                processed_files.add(fname)
                print(f"✓ Successfully processed and logged: {fname}")
            except Exception as e:
                print(f"✗ Processing failed (not logged): {e}")
        else:
            print(f"[Skipping] {fname} already has a resolution of {height}p")
            append_processed_log(log_path, fname)
            processed_files.add(fname)

if __name__ == "__main__":
    main()
