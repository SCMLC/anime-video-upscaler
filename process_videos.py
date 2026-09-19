import os
import sys
import urllib.request
import subprocess
import cv2
import numpy as np
import torch
import torchvision

# 修正 PyTorch 2.x 相容性
if not hasattr(torch, 'torchversion'):
    torch.torchversion = torchvision.__version__

# 開啟 CUDA 捲積與記憶體配置最佳化
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from tqdm import tqdm

# =========================================================================
# 💡 參數設定區
# =========================================================================
BATCH_SIZE = 4                  # 一次送入 GPU 推論的畫格數量
TARGET_HEIGHT = 720            # 最終目標高度 (720p)

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.rmvb', '.webm', '.ts')
LEGACY_EXTENSIONS = ('.rmvb', '.flv', '.wmv', '.avi')
LOG_FILENAME = "processed_files.txt"

MODEL_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'
MODEL_NAME = 'realesr-animevideov3.pth'

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
        print(f"正在下載動漫影片專用 AI 模型 ({MODEL_NAME})...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
        print("下載完成！")

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
    單次 4x 推理 + 純 GPU 顯存降採樣至 720p (防遠景小臉毀容)
    """
    if not batch_frames:
        return []

    # 1. 轉為 CUDA Tensor
    imgs = [f[:, :, ::-1] for f in batch_frames]
    imgs_array = np.stack(imgs, axis=0)
    
    curr_tensor = torch.from_numpy(np.ascontiguousarray(imgs_array.transpose(0, 3, 1, 2))).to(
        upsampler.device, non_blocking=True
    ).float() / 255.0

    if getattr(upsampler, 'half', False):
        curr_tensor = curr_tensor.half()

    model = upsampler.model
    model.eval()

    # 計算目標 720p 的尺寸 (維持原長寬比)
    _, _, orig_h, orig_w = curr_tensor.shape
    scale = target_h / orig_h
    target_w = int(orig_w * scale)
    if target_w % 2 != 0:
        target_w += 1

    # 2. 單次 4x 推理 + GPU 直接 Bicubic 下採樣 (平滑器官線條)
    with torch.no_grad():
        out_4x = model(curr_tensor)

        final_tensor = torch.nn.functional.interpolate(
            out_4x,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        ).clamp_(0, 1)

    # 3. 處理完畢後才拉回 CPU 轉給 FFmpeg
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

    print(f"\n[動漫 AI 單次 4x -> 720p 下採樣修復] {os.path.basename(input_path)}")
    print(f"解析度變更: {height}p -> 降採樣 {TARGET_HEIGHT}p (Batch Size = {BATCH_SIZE})")

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

    # 💡 調整 NVENC 參數：平衡畫質與檔案大小 (~300MB)
    ffmpeg_out_cmd = [
        'ffmpeg', '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{new_width}x{new_height}',
        '-pix_fmt', 'bgr24',
        '-r', f"{fps:.3f}",
        '-i', '-',
        '-i', input_path,
        '-map', '0:v:0',
        '-map', '1:a?',
        '-c:v', 'h264_nvenc',
        '-preset', 'p4',            # P4 平衡編碼效率與速度
        '-rc', 'vbr',
        '-cq', '23',                # CQ 23：品質優秀且容量控制良好
        '-maxrate', '3M',          # 限制最大位元率 3 Mbps (防止容量暴增)
        '-bufsize', '6M',
    ] + audio_codec_args + [output_path]
    # 💡 容量精確控制版 (目標 ~280MB)
    ffmpeg_out_cmd = [
        'ffmpeg', '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{new_width}x{new_height}',
        '-pix_fmt', 'bgr24',
        '-r', f"{fps:.3f}",
        '-i', '-',
        '-i', input_path,
        '-map', '0:v:0',
        '-map', '1:a?',
        '-c:v', 'h264_nvenc',
        '-preset', 'p4',
        '-rc', 'vbr',
        '-cq', '26',                # 改為 26 (大幅降低動漫畫面碼率，視覺幾乎無損)
        '-maxrate', '2.2M',         # 碼率上限壓至 2.2 Mbps
        '-bufsize', '4.4M',
    ] + audio_codec_args + [output_path]

    # 💡 修正色彩空間相容性 (yuv420p) + 控制容量的最佳化設定
    ffmpeg_out_cmd = [
        'ffmpeg', '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{new_width}x{new_height}',
        '-pix_fmt', 'bgr24',        # 輸入端：吃 Python 送進來的 BGR Raw Data
        '-r', f"{fps:.3f}",
        '-i', '-',
        '-i', input_path,
        '-map', '0:v:0',
        '-map', '1:a?',
        '-c:v', 'h264_nvenc',
        '-pix_fmt', 'yuv420p',      # 輸出端：強制轉為標準 yuv420p (解決播放相容性 + 降低容量)
        '-preset', 'p4',
        '-rc', 'vbr',
        '-cq', '26',
        '-maxrate', '2.2M',
        '-bufsize', '4.4M',
    ] + audio_codec_args + [output_path]

    proc_in = subprocess.Popen(ffmpeg_in_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    proc_out = subprocess.Popen(ffmpeg_out_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="AI 處理進度")
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
                print(f"\n[FFmpeg 錯誤]:\n{stderr_data.decode('utf-8', errors='ignore')}")
                raise RuntimeError("FFmpeg 進程異常終止")

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
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
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
        print("請提供目標影片資料夾路徑！")
        sys.exit(1)

    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f"指定路徑不存在: {target_dir}")
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
            print(f"[已紀錄處理過，跳過] {fname}")
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
                print(f"✓ 已成功轉檔並紀錄至 Log: {fname}")
            except Exception as e:
                print(f"✗ 處理失敗 (不記錄至 log): {e}")
        else:
            print(f"[跳過] {fname} 解析度已有 {height}p")
            append_processed_log(log_path, fname)
            processed_files.add(fname)

if __name__ == "__main__":
    main()
