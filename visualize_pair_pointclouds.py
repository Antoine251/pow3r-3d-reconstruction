#!/usr/bin/env python3
# Copyright (C) 2025-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# Gradio webapp to visualize each pair's point cloud after Pow3R inference.
# Helps debug per-pair outputs: pred1 pts3d (view1) vs pred2 pts3d_in_other_view (view2 in view1 frame).

import os
import argparse
import copy
import tempfile

# Import pandas before plotly to avoid circular import in plotly's color validator
import pandas  # noqa: F401

import cv2
import torch
import numpy as np
import plotly.graph_objects as go
import gradio as gr

import pow3r.tools.path_to_dust3r
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy, todevice, to_cpu, collate_with_cat
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from pow3r.model import Pow3R  # noqa: F401 - needed for eval(ckpt['definition'])

from demo_pow3r_multi_view import (
    load_pow3r_model,
    load_calibration,
    collect_images,
    undistort_images,
    build_calibration_for_images,
    pow3r_inference_with_calibration,
    get_3D_model_from_scene,
)

KEEP_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def print_pair_analysis(image_dir, calibration, scenegraph_type='complete', symmetrize=True, image_size=512):
    """Print for each pair: view1, view2, and relative pose (cam2 in cam1 frame) from calibration."""
    if calibration is None:
        print('>> --print_pair_analysis requires --calibration')
        return

    cameras, poses_c2w = load_calibration(calibration)
    if not poses_c2w:
        print('>> No camera poses in calibration')
        return
    image_files, tmpdir = undistort_images(
        collect_images(image_dir), cameras
    )
    K_list, pose_list, cam_order = build_calibration_for_images(
        image_files, cameras, poses_c2w, image_size
    )
    if pose_list is None:
        print('>> Could not build pose list from calibration')
        return

    imgs = load_images(image_files, size=image_size, verbose=False)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=symmetrize)
    print(f'\n>> Pair analysis (calibration-based, {len(pairs)} pairs)')
    print('   Each pair (view1, view2): output is in view1\'s frame.')
    print('   "cam2_in_cam1" = position of cam2 origin in cam1\'s frame (X=right, Y=up, Z=forward).')
    print('   "dir" = dominant direction from cam1 toward cam2.\n')

    by_dir = {}
    for idx, (v1, v2) in enumerate(pairs):
        i = _to_int(v1['idx'])
        j = _to_int(v2['idx'])
        cam1_name = cam_order[i] if i < len(cam_order) else f'img{i}'
        cam2_name = cam_order[j] if j < len(cam_order) else f'img{j}'

        c2w_1 = pose_list[i]
        c2w_2 = pose_list[j]
        R1 = c2w_1[:3, :3]
        t1 = c2w_1[:3, 3]
        t2 = c2w_2[:3, 3]
        # cam2 origin in cam1 frame: R1^T @ (t2 - t1)
        cam2_in_cam1 = R1.T @ (t2 - t1)
        dist = np.linalg.norm(cam2_in_cam1)

        # Dominant direction (X, Y, Z in cam1 frame)
        ax = np.argmax(np.abs(cam2_in_cam1))
        sgn = 1 if cam2_in_cam1[ax] > 0 else -1
        dir_names = ['X', 'Y', 'Z']
        dir_labels = {0: ('right', 'left'), 1: ('up', 'down'), 2: ('forward', 'back')}
        d = dir_labels[ax][0] if sgn > 0 else dir_labels[ax][1]
        key = d
        by_dir.setdefault(key, []).append((idx, i, j))

        print(f'  Pair #{idx:2d}: {i}-{j}  ({cam1_name} -> {cam2_name})')
        print(f'           view1={i}, view2={j}  |  cam2_in_cam1 = [{cam2_in_cam1[0]:+.4f}, {cam2_in_cam1[1]:+.4f}, {cam2_in_cam1[2]:+.4f}]  dist={dist:.4f}')
        print(f'           dominant dir: {d} ({dir_names[ax]}{"+" if sgn > 0 else "-"})')
        print()

    print('>> Pairs grouped by dominant direction (cam2 relative to cam1):')
    for d in ['right', 'left', 'up', 'down', 'forward', 'back']:
        if d in by_dir:
            pairs_str = ', '.join(f'{i}-{j}' for _, i, j in by_dir[d])
            print(f'   {d}: {pairs_str}')

    if tmpdir:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _pair_idx(v1, v2):
    try:
        i = int(v1['idx'][0])
    except (TypeError, IndexError):
        i = int(v1['idx'])
    try:
        j = int(v2['idx'][0])
    except (TypeError, IndexError):
        j = int(v2['idx'])
    return i, j


def run_inference(image_dir, calibration, model, device, image_size=512,
                  scenegraph_type='complete', symmetrize=True):
    """Run Pow3R inference and return output (view1, view2, pred1, pred2)."""
    image_files = collect_images(image_dir)
    if len(image_files) < 2:
        raise RuntimeError(f'Need at least 2 images, found {len(image_files)} in {image_dir}')

    calib_data = None
    tmpdir = None
    if calibration is not None:
        cameras, poses_c2w = load_calibration(calibration)
        print('>> Undistorting images...')
        image_files, tmpdir = undistort_images(image_files, cameras)
        K_list, pose_list, cam_order = build_calibration_for_images(
            image_files, cameras, poses_c2w, image_size)
        if K_list is not None:
            calib_data = {'K_list': K_list, 'pose_list': pose_list, 'cam_order': cam_order}
            print(f'>> Camera assignment: {cam_order}')

    imgs = load_images(image_files, size=image_size, verbose=True)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    print(f'>> Loaded {len(imgs)} images')

    if tmpdir is not None:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=symmetrize)
    keep_set = set(KEEP_PAIRS)
    pairs = [(v1, v2) for v1, v2 in pairs if _pair_idx(v1, v2) in keep_set]
    pair_strs = [f'{i}-{j}' for v1, v2 in pairs for i, j in [_pair_idx(v1, v2)]]
    print(f'>> Created {len(pairs)} image pairs' + (' (symmetrized)' if symmetrize else '') + f': {", ".join(pair_strs)}')

    if len(pairs) == 0:
        raise RuntimeError(
            f'No pairs created. Try --scenegraph_type complete (supported: complete, swin, oneref).'
        )

    if calib_data is not None:
        output = pow3r_inference_with_calibration(
            pairs, model, device, calib_data, batch_size=1, verbose=True
        )
    else:
        output = inference(pairs, model, device, batch_size=1, verbose=True)

    return output, imgs, calib_data


def run_full_pipeline(image_dir, calibration, model, device, outdir='output',
                      image_size=512, scenegraph_type='complete', symmetrize=True):
    """Run inference + global alignment and export scene.glb."""
    output, imgs, calib_data = run_inference(
        image_dir, calibration, model, device,
        image_size=image_size,
        scenegraph_type=scenegraph_type,
        symmetrize=symmetrize,
    )
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    print(f'>> Global alignment mode: {mode.value}')
    scene = global_aligner(output, device=device, mode=mode, verbose=True)

    if calib_data is not None and mode == GlobalAlignerMode.PointCloudOptimizer:
        K_list = calib_data['K_list']
        pose_list = calib_data['pose_list']
        n_cameras = len(set(calib_data['cam_order']))
        n_images = len(imgs)
        known_focals = [float(np.mean([K[0, 0], K[1, 1]])) for K in K_list]
        known_pp = [K[:2, 2] for K in K_list]
        print(f'>> Presetting {n_images} focal lengths and principal points from calibration')
        scene.preset_focal(known_focals)
        for i, pp in enumerate(known_pp):
            H, W = scene.imshapes[i]
            # PointCloudOptimizer stores principal point offsets in pixel/10 units.
            scene.im_pp[i].data[:] = (torch.tensor(pp, dtype=torch.float32) - torch.tensor([W / 2, H / 2])) / 10.0
        if pose_list is not None and n_images == n_cameras:
            known_poses = [torch.tensor(p, dtype=torch.float32) for p in pose_list]
            print(f'>> Presetting {n_images} camera poses from calibration')
            scene.preset_pose(known_poses)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        lr = 0.01
        # Use MST init even with calibration to avoid known_poses depth init artifacts.
        init_mode = 'known_poses'
        loss = scene.compute_global_alignment(init=init_mode, niter=300, schedule='linear', lr=lr)
        print(f'>> Global alignment final loss: {loss}')

    cams2world = to_numpy(scene.get_im_poses().cpu())
    print(f'\n>> Camera positions after alignment:')
    for i, pose in enumerate(cams2world):
        pos = pose[:3, 3]
        print(f'   cam #{i}: x={pos[0]:.6f}, y={pos[1]:.6f}, z={pos[2]:.6f}')

    print(f'\n>> Pairwise distances:')
    for i in range(len(cams2world)):
        for j in range(i + 1, len(cams2world)):
            dist = np.linalg.norm(cams2world[i][:3, 3] - cams2world[j][:3, 3])
            print(f'   cam #{i} <-> cam #{j}: {dist:.6f}')

    os.makedirs(outdir, exist_ok=True)
    glb_path = get_3D_model_from_scene(
        outdir, scene,
        min_conf_thr=3,
        as_pointcloud=False,
        mask_sky=False,
        clean_depth=True,
        cam_size=0.01,
    )
    print(f'>> Exported scene to {glb_path}')
    return output, imgs, calib_data, glb_path


def extract_pair_data(output, pair_idx):
    """Extract pred1, pred2 for a single pair from collated output."""
    pred1 = output['pred1']
    pred2 = output['pred2']

    def sel(d, i):
        if isinstance(d, dict):
            return {k: sel(v, i) for k, v in d.items()}
        if isinstance(d, torch.Tensor):
            return d[i]
        if isinstance(d, (list, tuple)):
            return d[i]
        return d

    return sel(pred1, pair_idx), sel(pred2, pair_idx)


def _to_int(x):
    """Extract int from scalar, list, array, or tensor."""
    if isinstance(x, (list, tuple)):
        return int(x[0]) if x else 0
    if isinstance(x, np.ndarray):
        return int(x.flat[0])
    if isinstance(x, torch.Tensor):
        return int(x.cpu().numpy().flat[0])
    return int(x)


def get_pair_indices(output):
    """Get (i, j) view indices for each pair."""
    view1 = output['view1']
    view2 = output['view2']
    idx1 = view1['idx']
    idx2 = view2['idx']
    # collate_with_cat(lists=True) yields a list of tensors, not a stacked tensor
    if isinstance(idx1, (list, tuple)):
        n_pairs = len(idx1)
        pairs = [(_to_int(idx1[e]), _to_int(idx2[e])) for e in range(n_pairs)]
    elif isinstance(idx1, torch.Tensor):
        idx1 = idx1.cpu().numpy()
        idx2 = idx2.cpu().numpy()
        n_pairs = idx1.shape[0] if idx1.ndim > 0 else 1
        pairs = [(_to_int(idx1[e]), _to_int(idx2[e])) for e in range(n_pairs)]
    else:
        pairs = [(_to_int(idx1), _to_int(idx2))]
    return pairs


def get_pair_images(output, pair_idx):
    """Get RGB images for view1 and view2 of a pair."""
    view1 = output['view1']
    view2 = output['view2']

    def sel(d, i):
        if isinstance(d, dict):
            return {k: sel(v, i) for k, v in d.items()}
        if isinstance(d, torch.Tensor):
            return d[i]
        if isinstance(d, (list, tuple)):
            return d[i]
        return d

    v1 = sel(view1, pair_idx)
    v2 = sel(view2, pair_idx)
    img1 = rgb(v1['img']) if 'img' in v1 else None
    img2 = rgb(v2['img']) if 'img' in v2 else None
    if isinstance(img1, list):
        img1 = img1[0] if img1 else None
    if isinstance(img2, list):
        img2 = img2[0] if img2 else None
    return img1, img2


def pts3d_to_flat_valid(pts3d, conf, conf_thr=1.0, subsample=4, img=None):
    """Flatten pts3d (H,W,3) to (N,3), filter by conf, optionally subsample.
    If img (H,W,3) is provided, also return RGB colors for each point."""
    pts3d = to_numpy(pts3d)
    conf = to_numpy(conf)
    if pts3d.ndim == 4:
        pts3d = pts3d.squeeze(0)
    if conf.ndim == 3:
        conf = conf.squeeze(0)
    valid = np.isfinite(pts3d).all(axis=-1) & (conf >= conf_thr)
    pts_flat = pts3d.reshape(-1, 3)
    valid_flat = valid.ravel()
    pts_valid = pts_flat[valid_flat]
    colors_valid = None
    if img is not None:
        img = to_numpy(img) if hasattr(img, 'cpu') else np.asarray(img)
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)  # (3,H,W) -> (H,W,3)
        colors_flat = img.reshape(-1, 3)[valid_flat]
        colors_valid = colors_flat
    if subsample > 1 and len(pts_valid) > 0:
        idx = np.arange(len(pts_valid))
        np.random.seed(42)
        np.random.shuffle(idx)
        keep = idx[::subsample]
        pts_valid = pts_valid[keep]
        if colors_valid is not None:
            colors_valid = colors_valid[keep]
    return pts_valid, colors_valid


def build_plots(output, pair_idx, conf_thr=1.0, subsample=4, show_view1=True, show_view2=True):
    """Build two Plotly 3D figures: one RGB-colored, one red/blue by view."""
    pred1, pred2 = extract_pair_data(output, pair_idx)
    pairs = get_pair_indices(output)
    img1, img2 = get_pair_images(output, pair_idx)
    i, j = pairs[pair_idx]

    pts1, colors1 = pts3d_to_flat_valid(
        pred1['pts3d'], pred1['conf'],
        conf_thr=conf_thr, subsample=subsample, img=img1
    )
    pts2, colors2 = pts3d_to_flat_valid(
        pred2['pts3d_in_other_view'], pred2['conf'],
        conf_thr=conf_thr, subsample=subsample, img=img2
    )

    # Sanity check: print pts3d scale range before transform (helps debug per-camera scale mismatch)
    if len(pts1) > 0:
        d1 = np.linalg.norm(pts1, axis=1)
        print(f'Pair #{pair_idx} ({i}-{j}): pts1 (view{i}) range = [{d1.min():.4f}, {d1.max():.4f}]')
    if len(pts2) > 0:
        d2 = np.linalg.norm(pts2, axis=1)
        print(f'Pair #{pair_idx} ({i}-{j}): pts2 (view{j} in view{i}) range = [{d2.min():.4f}, {d2.max():.4f}]')

    # ---- Figure 1: RGB colored ----
    fig_rgb = go.Figure()
    if show_view1 and len(pts1) > 0:
        if colors1 is not None:
            # colors may be [0,1] or [0,255]
            scale = 255.0 if colors1.max() <= 1.0 else 1.0
            rgb_str = [f'rgb({int(np.clip(c[0]*scale,0,255))},{int(np.clip(c[1]*scale,0,255))},{int(np.clip(c[2]*scale,0,255))})' for c in colors1]
            fig_rgb.add_trace(go.Scatter3d(
                x=pts1[:, 0], y=pts1[:, 1], z=pts1[:, 2],
                mode='markers',
                marker=dict(size=2, color=rgb_str, opacity=0.9),
                name=f'View {i} (RGB)'
            ))
        else:
            fig_rgb.add_trace(go.Scatter3d(
                x=pts1[:, 0], y=pts1[:, 1], z=pts1[:, 2],
                mode='markers',
                marker=dict(size=2, color='gray', opacity=0.7),
                name=f'View {i}'
            ))
    if show_view2 and len(pts2) > 0:
        if colors2 is not None:
            scale = 255.0 if colors2.max() <= 1.0 else 1.0
            rgb_str = [f'rgb({int(np.clip(c[0]*scale,0,255))},{int(np.clip(c[1]*scale,0,255))},{int(np.clip(c[2]*scale,0,255))})' for c in colors2]
            fig_rgb.add_trace(go.Scatter3d(
                x=pts2[:, 0], y=pts2[:, 1], z=pts2[:, 2],
                mode='markers',
                marker=dict(size=2, color=rgb_str, opacity=0.9),
                name=f'View {j} (RGB)'
            ))
        else:
            fig_rgb.add_trace(go.Scatter3d(
                x=pts2[:, 0], y=pts2[:, 1], z=pts2[:, 2],
                mode='markers',
                marker=dict(size=2, color='gray', opacity=0.7),
                name=f'View {j}'
            ))
    fig_rgb.update_layout(
        title=f'Pair #{pair_idx}: views {i}–{j} — RGB colored',
        scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
    )

    # ---- Figure 2: Red / Blue by view ----
    fig_rb = go.Figure()
    if show_view1 and len(pts1) > 0:
        fig_rb.add_trace(go.Scatter3d(
            x=pts1[:, 0], y=pts1[:, 1], z=pts1[:, 2],
            mode='markers',
            marker=dict(size=2, color='red', opacity=0.7),
            name=f'View {i} pts3d'
        ))
    if show_view2 and len(pts2) > 0:
        fig_rb.add_trace(go.Scatter3d(
            x=pts2[:, 0], y=pts2[:, 1], z=pts2[:, 2],
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.7),
            name=f'View {j} pts3d_in_other_view'
        ))
    fig_rb.update_layout(
        title=f'Pair #{pair_idx}: views {i}–{j} — Red=view{i}, Blue=view{j}',
        scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
    )

    return fig_rgb, fig_rb


# Distinct colors for each pair in combined view (plotly qualitative palette)
PAIR_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]


def build_combined_plots(output, pair_choices, selected_pair_labels, conf_thr=1.0, subsample=4):
    """Build two figures with all selected pairs: one colored by pair, one RGB."""
    pairs = get_pair_indices(output)
    selected_indices = [pair_choices.index(lbl) for lbl in selected_pair_labels if lbl in pair_choices]

    fig_by_pair = go.Figure()
    fig_rgb = go.Figure()

    for k, pair_idx in enumerate(selected_indices):
        pred1, pred2 = extract_pair_data(output, pair_idx)
        img1, img2 = get_pair_images(output, pair_idx)
        i, j = pairs[pair_idx]

        pts1, colors1 = pts3d_to_flat_valid(
            pred1['pts3d'], pred1['conf'],
            conf_thr=conf_thr, subsample=subsample, img=img1
        )
        pts2, colors2 = pts3d_to_flat_valid(
            pred2['pts3d_in_other_view'], pred2['conf'],
            conf_thr=conf_thr, subsample=subsample, img=img2
        )

        # Sanity check: print pts3d scale range before transform (helps debug per-camera scale mismatch)
        if len(pts1) > 0:
            d1 = np.linalg.norm(pts1, axis=1)
            print(f'Pair #{pair_idx} ({i}-{j}): pts1 (view{i}) range = [{d1.min():.4f}, {d1.max():.4f}]')
        if len(pts2) > 0:
            d2 = np.linalg.norm(pts2, axis=1)
            print(f'Pair #{pair_idx} ({i}-{j}): pts2 (view{j} in view{i}) range = [{d2.min():.4f}, {d2.max():.4f}]')

        color = PAIR_COLORS[k % len(PAIR_COLORS)]
        name = f'Pair #{pair_idx}: {i}–{j}'

        # By-pair colored plot
        if len(pts1) > 0:
            fig_by_pair.add_trace(go.Scatter3d(
                x=pts1[:, 0], y=pts1[:, 1], z=pts1[:, 2],
                mode='markers',
                marker=dict(size=2, color=color, opacity=0.7),
                name=f'{name} view{i}'
            ))
        if len(pts2) > 0:
            fig_by_pair.add_trace(go.Scatter3d(
                x=pts2[:, 0], y=pts2[:, 1], z=pts2[:, 2],
                mode='markers',
                marker=dict(size=2, color=color, opacity=0.7),
                name=f'{name} view{j}'
            ))

        # RGB plot
        if len(pts1) > 0 and colors1 is not None:
            scale = 255.0 if colors1.max() <= 1.0 else 1.0
            rgb_str = [f'rgb({int(np.clip(c[0]*scale,0,255))},{int(np.clip(c[1]*scale,0,255))},{int(np.clip(c[2]*scale,0,255))})' for c in colors1]
            fig_rgb.add_trace(go.Scatter3d(
                x=pts1[:, 0], y=pts1[:, 1], z=pts1[:, 2],
                mode='markers',
                marker=dict(size=2, color=rgb_str, opacity=0.9),
                name=f'{name} view{i}'
            ))
        elif len(pts1) > 0:
            fig_rgb.add_trace(go.Scatter3d(
                x=pts1[:, 0], y=pts1[:, 1], z=pts1[:, 2],
                mode='markers',
                marker=dict(size=2, color='gray', opacity=0.7),
                name=f'{name} view{i}'
            ))
        if len(pts2) > 0 and colors2 is not None:
            scale = 255.0 if colors2.max() <= 1.0 else 1.0
            rgb_str = [f'rgb({int(np.clip(c[0]*scale,0,255))},{int(np.clip(c[1]*scale,0,255))},{int(np.clip(c[2]*scale,0,255))})' for c in colors2]
            fig_rgb.add_trace(go.Scatter3d(
                x=pts2[:, 0], y=pts2[:, 1], z=pts2[:, 2],
                mode='markers',
                marker=dict(size=2, color=rgb_str, opacity=0.9),
                name=f'{name} view{j}'
            ))
        elif len(pts2) > 0:
            fig_rgb.add_trace(go.Scatter3d(
                x=pts2[:, 0], y=pts2[:, 1], z=pts2[:, 2],
                mode='markers',
                marker=dict(size=2, color='gray', opacity=0.7),
                name=f'{name} view{j}'
            ))

    fig_by_pair.update_layout(
        title=f'Combined — colored by pair (select pairs below)',
        scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
    )
    fig_rgb.update_layout(
        title=f'Combined — RGB colored',
        scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
    )
    return fig_by_pair, fig_rgb


def main():
    parser = argparse.ArgumentParser(
        description='Visualize each pair point cloud after Pow3R inference (Gradio)'
    )
    parser.add_argument('--image_dir', type=str, default='./dataset/scene2',
                        help='Directory containing images')
    parser.add_argument('--ckpt_path', type=str,
                        default='models/Pow3R_ViTLarge_BaseDecoder_512_linear.pth',
                        help='Path to Pow3R checkpoint')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Path to calibration JSON')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--image_size', type=int, default=512, choices=[512, 224])
    parser.add_argument('--scenegraph_type', type=str, default='complete',
                        choices=['complete', 'swin', 'oneref'],
                        help='Scene graph: complete=all pairs, swin=sliding window, oneref=star from ref')
    parser.add_argument('--no_symmetrize', action='store_true',
                        help='Do not symmetrize pairs')
    parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    parser.add_argument('--server_port', type=int, default=None)
    parser.add_argument('--print_pair_analysis', action='store_true',
                        help='Print pair analysis (view1, view2, relative pose) and exit')
    parser.add_argument('--outdir', type=str, default='output', help='Output directory for scene.glb')
    args = parser.parse_args()

    if args.print_pair_analysis:
        print_pair_analysis(
            args.image_dir,
            args.calibration,
            scenegraph_type=args.scenegraph_type,
            symmetrize=not args.no_symmetrize,
            image_size=args.image_size,
        )
        return

    print('>> Loading Pow3R model...')
    model = load_pow3r_model(args.ckpt_path, device=args.device)

    print('>> Running full pipeline (inference + global alignment)...')
    output, imgs, calib_data, glb_path = run_full_pipeline(
        args.image_dir,
        args.calibration,
        model,
        args.device,
        outdir=args.outdir,
        image_size=args.image_size,
        scenegraph_type=args.scenegraph_type,
        symmetrize=not args.no_symmetrize,
    )

    pairs = get_pair_indices(output)
    n_pairs = len(pairs)
    pair_choices = [f'Pair #{i}: views {a}–{b}' for i, (a, b) in enumerate(pairs)]
    print(f'>> Inference done. {n_pairs} pairs: {pairs}')

    def update_display(pair_choice, conf_thr, subsample, show_view1, show_view2):
        pair_idx = pair_choices.index(pair_choice) if pair_choice in pair_choices else 0
        fig_rgb, fig_rb = build_plots(
            output, pair_idx,
            conf_thr=conf_thr,
            subsample=int(subsample),
            show_view1=show_view1,
            show_view2=show_view2,
        )
        img1, img2 = get_pair_images(output, pair_idx)
        return fig_rgb, fig_rb, img1, img2

    def update_combined(selected_labels, conf_thr, subsample):
        selected = list(selected_labels) if selected_labels else pair_choices
        if not selected:
            return go.Figure(), go.Figure()
        fig_bp, fig_rgb = build_combined_plots(
            output, pair_choices, selected,
            conf_thr=conf_thr,
            subsample=int(subsample),
        )
        return fig_bp, fig_rgb

    with gr.Blocks(title='Pow3R Pair Point Cloud Viewer') as demo:
        gr.Markdown('# Pow3R Pair Point Cloud Viewer')
        gr.Markdown(
            '**Tab 1**: Per-pair view. **Tab 2**: Combined view. **Tab 3**: Scene (GLB).'
        )

        with gr.Tabs():
            with gr.Tab('Per-pair view'):
                with gr.Row():
                    pair_dropdown = gr.Dropdown(
                        choices=pair_choices,
                        value=pair_choices[0],
                        label='Select pair',
                    )
                    conf_slider = gr.Slider(0.0, 20.0, value=1.0, step=0.5, label='Confidence threshold')
                    subsample_slider = gr.Slider(1, 16, value=4, step=1, label='Subsample (1=all)')
                with gr.Row():
                    show_view1 = gr.Checkbox(value=True, label='Show view1 pts3d')
                    show_view2 = gr.Checkbox(value=True, label='Show view2 pts3d_in_other_view')
                with gr.Row():
                    img1_out = gr.Image(label='View 1')
                    img2_out = gr.Image(label='View 2')
                with gr.Row():
                    plot_rgb = gr.Plot(label='RGB colored')
                    plot_rb = gr.Plot(label='Red / Blue by view')

                inputs = [pair_dropdown, conf_slider, subsample_slider, show_view1, show_view2]
                outputs = [plot_rgb, plot_rb, img1_out, img2_out]
                pair_dropdown.change(update_display, inputs=inputs, outputs=outputs)
                conf_slider.change(update_display, inputs=inputs, outputs=outputs)
                subsample_slider.change(update_display, inputs=inputs, outputs=outputs)
                show_view1.change(update_display, inputs=inputs, outputs=outputs)
                show_view2.change(update_display, inputs=inputs, outputs=outputs)

                demo.load(
                    fn=lambda: update_display(pair_choices[0], 1.0, 4, True, True),
                    outputs=outputs,
                )

            with gr.Tab('Combined view'):
                with gr.Row():
                    combined_conf = gr.Slider(0.0, 20.0, value=1.0, step=0.5, label='Confidence threshold')
                    combined_subsample = gr.Slider(1, 16, value=4, step=1, label='Subsample (1=all)')
                pair_checklist = gr.CheckboxGroup(
                    choices=pair_choices,
                    value=[pair_choices[0]] if pair_choices else [],
                    label='Select pairs to display (uncheck to hide)',
                )
                update_btn = gr.Button('Update plots', variant='primary')
                with gr.Row():
                    plot_combined_bp = gr.Plot(label='Colored by pair')
                    plot_combined_rgb = gr.Plot(label='RGB colored')

                combined_inputs = [pair_checklist, combined_conf, combined_subsample]
                combined_outputs = [plot_combined_bp, plot_combined_rgb]
                update_btn.click(update_combined, inputs=combined_inputs, outputs=combined_outputs)

                demo.load(
                    fn=lambda: update_combined([pair_choices[0]] if pair_choices else [], 1.0, 4),
                    outputs=combined_outputs,
                )

            with gr.Tab('Scene (GLB)'):
                gr.Markdown('Globally aligned 3D reconstruction.')
                gr.Model3D(
                    value=glb_path if glb_path and os.path.exists(glb_path) else None,
                    label='3D Scene',
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                )

    print('>> Launching Gradio app...')
    demo.launch(share=args.share, server_port=args.server_port)


if __name__ == '__main__':
    main()
