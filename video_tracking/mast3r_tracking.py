import os
import sys
import glob
import tempfile
import types
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for SSH/headless
from matplotlib import pyplot as pl

# Add mast3r and dust3r to path (same as demo_mast3r_multi_view.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAST3R_REPO = os.path.join(ROOT, 'mast3r')
DUST3R_REPO = os.path.join(ROOT, 'dust3r')
sys.path.insert(0, DUST3R_REPO)
sys.path.insert(0, MAST3R_REPO)
sys.modules['mast3r.utils.path_to_dust3r'] = types.ModuleType('mast3r.utils.path_to_dust3r')

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images

# Sample every N-th frame from the video (1 = every frame).
FRAME_STRIDE = 100
# How many 2D features to follow from the reference frame across the sampled frames.
N_TRACK_POINTS = 32
# Reject a match if the nearest correspondence on the reference image is farther than this (pixels).
TRACK_MATCH_MAX_DIST_PX = 24.0
# Inference batch size for (ref, target) pairs; lower if GPU runs out of memory.
INFERENCE_BATCH_SIZE = 4
# Output MP4: frames per second (one frame per sampled timestep).
TRACK_VIDEO_FPS = 8


def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return max(0, n)


def sampled_frame_indices(n_frames, stride, include_last=True):
    """Indices 0, stride, 2*stride, ... optionally forcing the last frame."""
    if n_frames <= 0:
        return []
    if stride < 1:
        raise ValueError("stride must be >= 1")
    idx = list(range(0, n_frames, stride))
    if include_last and idx and idx[-1] < n_frames - 1:
        idx.append(n_frames - 1)
    elif include_last and not idx:
        idx = [0]
    return idx


def filter_valid_matches(matches_im0, matches_im1, W0, H0, W1, H1, border=3):
    valid0 = (matches_im0[:, 0] >= border) & (matches_im0[:, 0] < int(W0) - border) & (
        matches_im0[:, 1] >= border) & (matches_im0[:, 1] < int(H0) - border)
    valid1 = (matches_im1[:, 0] >= border) & (matches_im1[:, 0] < int(W1) - border) & (
        matches_im1[:, 1] >= border) & (matches_im1[:, 1] < int(H1) - border)
    m = valid0 & valid1
    return matches_im0[m], matches_im1[m]


def true_hw(true_shape_entry):
    """One row from batched true_shape: (H, W) as ints."""
    if torch.is_tensor(true_shape_entry):
        t = true_shape_entry.detach().cpu().numpy()
    else:
        t = np.asarray(true_shape_entry)
    t = np.asarray(t).reshape(-1)
    if t.size < 2:
        raise ValueError(f"Bad true_shape: {true_shape_entry!r}")
    return int(t[0]), int(t[1])


def match_from_desc(desc1, desc2, H0, W0, H1, W1, device):
    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1, desc2, subsample_or_initxy1=8, device=device, dist='dot', block_size=2**13)
    return filter_valid_matches(matches_im0, matches_im1, W0, H0, W1, H1)


def pick_track_seeds(matches_im0, n_track):
    n = matches_im0.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if n_track >= n:
        return matches_im0.astype(np.float64)
    ix = np.round(np.linspace(0, n - 1, n_track)).astype(int)
    return matches_im0[ix].astype(np.float64)


def track_seeds_in_matches(seeds_ref, matches_im0, matches_im1, max_dist_px):
    """For each seed on the reference image, take the closest match correspondence."""
    if seeds_ref.size == 0 or matches_im0.shape[0] == 0:
        return np.full((seeds_ref.shape[0], 2), np.nan), np.zeros(seeds_ref.shape[0], dtype=bool)
    d = np.linalg.norm(matches_im0[np.newaxis, :, :] - seeds_ref[:, np.newaxis, :], axis=2)
    j = np.argmin(d, axis=1)
    min_d = d[np.arange(seeds_ref.shape[0]), j]
    ok = min_d <= max_dist_px
    pts = matches_im1[j].astype(np.float64)
    pts[~ok] = np.nan
    return pts, ok


def _rgb_float01_to_bgr_u8(rgb):
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _jet_bgr(fi, n_feat):
    cmap = pl.get_cmap('jet')
    rgba = cmap(fi / max(n_feat - 1, 1))
    return int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255)


def write_tracks_mp4(
        out_path, panels_rgb, frame_indices, seeds_ref, all_tracks, fps, title_prefix="MASt3R tracks"):
    """One video frame per entry: panel 0 = reference with seed markers; rest = targets with dots."""
    n_feat = seeds_ref.shape[0]
    max_h = max(p.shape[0] for p in panels_rgb)
    max_w = max(p.shape[1] for p in panels_rgb)
    size = (max_w, max_h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), size)
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {out_path}")

    try:
        for i, rgb in enumerate(panels_rgb):
            pad_h, pad_w = max_h - rgb.shape[0], max_w - rgb.shape[1]
            bgr = _rgb_float01_to_bgr_u8(rgb)
            if pad_h or pad_w:
                bgr = cv2.copyMakeBorder(
                    bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            fi_idx = frame_indices[i]
            line1 = f"{title_prefix} | frame {fi_idx} ({i + 1}/{len(panels_rgb)})"
            cv2.putText(
                bgr, line1, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            if i == 0:
                for s in range(n_feat):
                    x, y = int(round(seeds_ref[s, 0])), int(round(seeds_ref[s, 1]))
                    cv2.drawMarker(
                        bgr, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)
            else:
                tk = all_tracks[i - 1]
                for f in range(n_feat):
                    if np.any(np.isnan(tk[f])):
                        continue
                    x, y = int(round(tk[f, 0])), int(round(tk[f, 1]))
                    col = _jet_bgr(f, n_feat)
                    cv2.circle(bgr, (x, y), 4, col, -1, cv2.LINE_AA)
            writer.write(bgr)
    finally:
        writer.release()


def _squeeze_desc_batch(pred_desc, batch_idx):
    """One image descriptor map (H, W, C) for fast_reciprocal_NNs."""
    d = pred_desc[batch_idx]
    while d.ndim > 3:
        d = d.squeeze(0)
    if d.ndim != 3:
        raise ValueError(f"Expected desc (H, W, C), got shape {tuple(d.shape)}")
    return d


def extract_frames_from_video(video_path, frame_indices, crop_w, crop_h, offset_x, offset_y):
    """Extract frames from a video at the given indices, crop to crop_w x crop_h.
    offset_x, offset_y shift the crop from center (0,0 = center crop)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    temp_paths = []
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise IOError(f"Could not read frame {idx} from video (video may have fewer frames)")
            H, W = frame.shape[:2]
            # Crop center + offset (0,0 = middle of image)
            x1 = (W - crop_w) // 2 + offset_x
            y1 = (H - crop_h) // 2 + offset_y
            x1 = max(0, min(x1, W - crop_w))
            y1 = max(0, min(y1, H - crop_h))
            x2, y2 = x1 + crop_w, y1 + crop_h
            frame_cropped = frame[y1:y2, x1:x2]
            frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            pil_img.save(path)
            temp_paths.append(path)
    finally:
        cap.release()

    return temp_paths


if __name__ == '__main__':
    device = 'cuda'
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    # Crop parameters: crop to 512x384 directly (no resize)
    # OFFSET_X, OFFSET_Y shift the crop from center (0, 0 = middle of image)
    CROP_WIDTH = 512
    CROP_HEIGHT = 384
    OFFSET_X = 50
    OFFSET_Y = -70

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mp4_files = glob.glob(os.path.join(script_dir, "*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No mp4 video found in {script_dir}")
    video_path = mp4_files[0]

    n_frames = get_video_frame_count(video_path)
    frame_indices = sampled_frame_indices(n_frames, FRAME_STRIDE, include_last=True)
    if len(frame_indices) < 2:
        raise ValueError(
            f"Need at least 2 sampled frames (got {len(frame_indices)}). "
            f"Video has {n_frames} frames; try FRAME_STRIDE=1 or a shorter stride.")

    print(f"Video frames: {n_frames}, sampling every {FRAME_STRIDE} -> {len(frame_indices)} images "
          f"(indices {frame_indices[0]} … {frame_indices[-1]})")

    temp_paths = extract_frames_from_video(
        video_path, frame_indices, CROP_WIDTH, CROP_HEIGHT, OFFSET_X, OFFSET_Y)
    try:
        images = load_images(temp_paths, size=512)
    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    ref = images[0]
    pairs = [(ref, images[i]) for i in range(1, len(images))]
    output = inference(
        pairs, model, device, batch_size=min(INFERENCE_BATCH_SIZE, len(pairs)), verbose=False)

    view1, view2 = output['view1'], output['view2']
    pred1, pred2 = output['pred1'], output['pred2']
    n_pairs = len(pairs)

    # Batched tensors: batch index corresponds to pairs[batch_idx] == (ref, images[batch_idx + 1])
    all_tracks = []  # list length n_pairs; each (N_TRACK_POINTS, 2) in target image coords
    seeds_ref = None

    for k in range(n_pairs):
        desc1 = _squeeze_desc_batch(pred1['desc'].detach(), k)
        desc2 = _squeeze_desc_batch(pred2['desc'].detach(), k)
        H0, W0 = true_hw(view1['true_shape'][k])
        H1, W1 = true_hw(view2['true_shape'][k])
        m0, m1 = match_from_desc(desc1, desc2, H0, W0, H1, W1, device)
        if k == 0:
            seeds_ref = pick_track_seeds(m0, N_TRACK_POINTS)
            if seeds_ref.shape[0] == 0:
                raise RuntimeError("No matches on the first (ref, target) pair; cannot seed tracks.")
            print(f"Tracking {seeds_ref.shape[0]} features from reference frame (sample index 0).")
        pts, ok = track_seeds_in_matches(seeds_ref, m0, m1, TRACK_MATCH_MAX_DIST_PX)
        all_tracks.append(pts)
        if not np.all(ok):
            n_bad = np.sum(~ok)
            print(f"  pair {k} (frame {frame_indices[k + 1]}): {n_bad}/{len(ok)} tracks exceeded "
                  f"match distance ({TRACK_MATCH_MAX_DIST_PX}px)")

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    def view_to_rgb(view, batch_idx=0):
        img_t = view['img']
        if img_t.ndim == 4:
            img_t = img_t[batch_idx] if img_t.shape[0] > 1 else img_t.squeeze(0)
        rgb_tensor = img_t * image_std.squeeze(0) + image_mean.squeeze(0)
        return rgb_tensor.permute(1, 2, 0).cpu().numpy()

    # --- Video: one frame per sampled timestep (ref, then each target with tracks) ---
    ref_rgb = view_to_rgb(view1, 0)
    panels = [ref_rgb]
    for k in range(n_pairs):
        panels.append(view_to_rgb(view2, k))

    n_feat = seeds_ref.shape[0]
    out_video = os.path.join(script_dir, 'mast3r_tracks.mp4')
    write_tracks_mp4(
        out_video, panels, frame_indices, seeds_ref, all_tracks, TRACK_VIDEO_FPS,
        title_prefix=f"stride={FRAME_STRIDE} | {n_feat} pts")
    print(f"Saved track video to {out_video} ({TRACK_VIDEO_FPS} fps, {len(panels)} frames)")

    # --- Optional: same as original two-panel match lines for first pair only ---
    k0 = 0
    desc1 = _squeeze_desc_batch(pred1['desc'].detach(), k0)
    desc2 = _squeeze_desc_batch(pred2['desc'].detach(), k0)
    h0a, w0a = true_hw(view1['true_shape'][k0])
    h1a, w1a = true_hw(view2['true_shape'][k0])
    matches_im0, matches_im1 = match_from_desc(
        desc1, desc2, h0a, w0a, h1a, w1a, device)
    n_viz = min(50, matches_im0.shape[0])
    if n_viz > 0:
        match_idx_to_viz = np.round(
            np.linspace(0, matches_im0.shape[0] - 1, n_viz)).astype(int)
        viz_matches_im0 = matches_im0[match_idx_to_viz]
        viz_matches_im1 = matches_im1[match_idx_to_viz]
        viz_imgs = [view_to_rgb(view1, k0), view_to_rgb(view2, k0)]
        H0b, W0b, H1b, W1b = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
        img0 = np.pad(viz_imgs[0], ((0, max(H1b - H0b, 0)), (0, 0), (0, 0)), constant_values=0)
        img1 = np.pad(viz_imgs[1], ((0, max(H0b - H1b, 0)), (0, 0), (0, 0)), constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        pl.figure()
        pl.imshow(img)
        cmap_m = pl.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            pl.plot(
                [x0, x1 + W0b], [y0, y1], '-+',
                color=cmap_m(i / max(n_viz - 1, 1)), scalex=False, scaley=False)
        out_pair = os.path.join(script_dir, 'mast3r_matches.png')
        pl.savefig(out_pair, dpi=150, bbox_inches='tight')
        pl.close()
        print(f"Saved first-pair matches visualization to {out_pair}")