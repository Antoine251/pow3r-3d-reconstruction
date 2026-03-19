# Copyright (C) 2025-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# Multi-view 3D reconstruction using POW3R, inspired by DUSt3R's demo pipeline.
# Runs POW3R inference on all image pairs, then performs global alignment
# to produce a coherent 3D reconstruction from multiple views.
# Supports optional camera calibration (intrinsics, distortion, extrinsic poses).

import os
import re
import json
import argparse
import copy
import tempfile

import torch
from pow3r.model.inference import AsymmetricSliding

import subprocess
import sys
from pow3r.model.inference import AsymmetricSliding
#from dust3r.inference import inference

import cv2
import numpy as np
import trimesh
import matplotlib.pyplot as plt
#from scipy.spatial.transform import Rotation

#import pow3r.tools.path_to_dust3r
#from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.utils.device import todevice
from dust3r.viz import add_scene_cam, CAM_COLORS #, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

#from pow3r.model import Pow3R  # noqa: F401 - needed for eval(ckpt['definition'])

# import plotly.graph_objects as go
# import gradio as gr


def load_pow3r_model(ckpt_path, device='cpu'):
    """Load the base POW3R model from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_def = ckpt['definition']
    print(f'>> Creating POW3R model = {model_def}')
    model = eval(model_def)
    print(f'>> Loading weights:', model.load_state_dict(ckpt['weights']))
    return model.to(device).eval()

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


# def build_plots(output, pair_idx, conf_thr=1.0, subsample=4, show_view1=True, show_view2=True, pose_list=None):
#     """Build two Plotly 3D figures: one RGB-colored, one red/blue by view.
#     If pose_list is provided (calibration), transform all pts to cam0 frame for consistent orientation."""
#     pred1, pred2 = extract_pair_data(output, pair_idx)
#     pairs = get_pair_indices(output)
#     img1, img2 = get_pair_images(output, pair_idx)
#     i, j = pairs[pair_idx]

#     pts1, colors1 = pts3d_to_flat_valid(
#         pred1['pts3d'], pred1['conf'],
#         conf_thr=conf_thr, subsample=subsample, img=img1
#     )
#     pts2, colors2 = pts3d_to_flat_valid(
#         pred2['pts3d_in_other_view'], pred2['conf'],
#         conf_thr=conf_thr, subsample=subsample, img=img2
#     )

#     # Sanity check: print pts3d scale range before transform (helps debug per-camera scale mismatch)
#     if len(pts1) > 0:
#         d1 = np.linalg.norm(pts1, axis=1)
#         print(f'Pair #{pair_idx} ({i}-{j}): pts1 (view{i}) range = [{d1.min():.4f}, {d1.max():.4f}]')
#     if len(pts2) > 0:
#         d2 = np.linalg.norm(pts2, axis=1)
#         print(f'Pair #{pair_idx} ({i}-{j}): pts2 (view{j} in view{i}) range = [{d2.min():.4f}, {d2.max():.4f}]')

#     if pose_list is not None:
#         pts1 = transform_pts_to_ref_frame(pts1, i, pose_list)
#         pts2 = transform_pts_to_ref_frame(pts2, i, pose_list)

#     frame_note = ' (cam0 frame)' if pose_list is not None else ''

#     # ---- Figure 1: RGB colored ----
#     fig_rgb = go.Figure()
#     if show_view1 and len(pts1) > 0:
#         if colors1 is not None:
#             # colors may be [0,1] or [0,255]
#             scale = 255.0 if colors1.max() <= 1.0 else 1.0
#             rgb_str = [f'rgb({int(np.clip(c[0]*scale,0,255))},{int(np.clip(c[1]*scale,0,255))},{int(np.clip(c[2]*scale,0,255))})' for c in colors1]
#             fig_rgb.add_trace(go.Scatter3d(
#                 x=pts1[:, 0], y=pts1[:, 1], z=pts1[:, 2],
#                 mode='markers',
#                 marker=dict(size=2, color=rgb_str, opacity=0.9),
#                 name=f'View {i} (RGB)'
#             ))
#         else:
#             fig_rgb.add_trace(go.Scatter3d(
#                 x=pts1[:, 0], y=pts1[:, 1], z=pts1[:, 2],
#                 mode='markers',
#                 marker=dict(size=2, color='gray', opacity=0.7),
#                 name=f'View {i}'
#             ))
#     if show_view2 and len(pts2) > 0:
#         if colors2 is not None:
#             scale = 255.0 if colors2.max() <= 1.0 else 1.0
#             rgb_str = [f'rgb({int(np.clip(c[0]*scale,0,255))},{int(np.clip(c[1]*scale,0,255))},{int(np.clip(c[2]*scale,0,255))})' for c in colors2]
#             fig_rgb.add_trace(go.Scatter3d(
#                 x=pts2[:, 0], y=pts2[:, 1], z=pts2[:, 2],
#                 mode='markers',
#                 marker=dict(size=2, color=rgb_str, opacity=0.9),
#                 name=f'View {j} (RGB)'
#             ))
#         else:
#             fig_rgb.add_trace(go.Scatter3d(
#                 x=pts2[:, 0], y=pts2[:, 1], z=pts2[:, 2],
#                 mode='markers',
#                 marker=dict(size=2, color='gray', opacity=0.7),
#                 name=f'View {j}'
#             ))
#     fig_rgb.update_layout(
#         title=f'Pair #{pair_idx}: views {i}–{j} — RGB colored{frame_note}',
#         scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
#         margin=dict(l=0, r=0, t=50, b=0),
#         showlegend=True,
#     )

#     # ---- Figure 2: Red / Blue by view ----
#     fig_rb = go.Figure()
#     if show_view1 and len(pts1) > 0:
#         fig_rb.add_trace(go.Scatter3d(
#             x=pts1[:, 0], y=pts1[:, 1], z=pts1[:, 2],
#             mode='markers',
#             marker=dict(size=2, color='red', opacity=0.7),
#             name=f'View {i} pts3d'
#         ))
#     if show_view2 and len(pts2) > 0:
#         fig_rb.add_trace(go.Scatter3d(
#             x=pts2[:, 0], y=pts2[:, 1], z=pts2[:, 2],
#             mode='markers',
#             marker=dict(size=2, color='blue', opacity=0.7),
#             name=f'View {j} pts3d_in_other_view'
#         ))
#     fig_rb.update_layout(
#         title=f'Pair #{pair_idx}: views {i}–{j} — Red=view{i}, Blue=view{j}{frame_note}',
#         scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
#         margin=dict(l=0, r=0, t=50, b=0),
#         showlegend=True,
#     )

#     return fig_rgb, fig_rb

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
    try:
        if hasattr(x, "item"):
            return int(x.item())
        return int(x)
    except Exception:
        return int(x)

# def get_pair_indices(output):
#     """Get (i, j) view indices for each pair."""
#     view1 = output['view1']
#     view2 = output['view2']
#     idx1 = view1['idx']
#     idx2 = view2['idx']
#     # collate_with_cat(lists=True) yields a list of tensors, not a stacked tensor
#     if isinstance(idx1, (list, tuple)):
#         n_pairs = len(idx1)
#         pairs = [(_to_int(idx1[e]), _to_int(idx2[e])) for e in range(n_pairs)]
#     elif isinstance(idx1, torch.Tensor):
#         idx1 = idx1.cpu().numpy()
#         idx2 = idx2.cpu().numpy()
#         n_pairs = idx1.shape[0] if idx1.ndim > 0 else 1
#         pairs = [(_to_int(idx1[e]), _to_int(idx2[e])) for e in range(n_pairs)]
#     else:
#         pairs = [(_to_int(idx1), _to_int(idx2))]
#     return pairs


# def get_pair_images(output, pair_idx):
#     """Get RGB images for view1 and view2 of a pair."""
#     view1 = output['view1']
#     view2 = output['view2']

#     def sel(d, i):
#         if isinstance(d, dict):
#             return {k: sel(v, i) for k, v in d.items()}
#         if isinstance(d, torch.Tensor):
#             return d[i]
#         if isinstance(d, (list, tuple)):
#             return d[i]
#         return d

#     v1 = sel(view1, pair_idx)
#     v2 = sel(view2, pair_idx)
#     img1 = rgb(v1['img']) if 'img' in v1 else None
#     img2 = rgb(v2['img']) if 'img' in v2 else None
#     if isinstance(img1, list):
#         img1 = img1[0] if img1 else None
#     if isinstance(img2, list):
#         img2 = img2[0] if img2 else None
#     return img1, img2

def load_segmentation_masks(folder_path):

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    mask_files = sorted([
        f for f in os.listdir(folder_path)
        if "mask" in f.lower() and f.lower().endswith(exts)
    ])

    if not mask_files:
        print(f"Warning: No files containing 'mask' found in {folder_path}")
        return None

    my_binary_masks = []
    
    for filename in mask_files:
        full_path = os.path.join(folder_path, filename)
        
        mask_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        
        if mask_img is None:
            print(f"Skipping {filename}: Could not read image.")
            continue
            
        _, binary_mask = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)
        
        my_binary_masks.append(binary_mask)

    print(f"Loaded {len(my_binary_masks)} segmentation masks.")
    return my_binary_masks

def load_calibration(calib_path):
    with open(calib_path) as f:
        calib = json.load(f)

    cameras = {}
    for cam_name, cam_data in calib['cameras'].items():
        dist = np.array(cam_data['dist'], dtype=np.float64)
        if dist.ndim > 1:
            dist = dist.flatten()

        cameras[cam_name] = {
            'K': np.array(cam_data['K'], dtype=np.float64),
            'dist': dist,
            'image_size': tuple(cam_data['image_size']),
        }

    poses_c2w = {}
    for pose_name, pose_data in calib['camera_poses'].items():
        R = np.array(pose_data['R'], dtype=np.float64)
        T = np.array(pose_data['T'], dtype=np.float64)

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ T
        if '_to_' not in pose_name:
            cam_name = pose_name
        else:
            cam_name = pose_name.split('_to_')[0]
        poses_c2w[cam_name] = c2w

    n_cameras = len(cameras)
    print(f'>> Loaded calibration: {n_cameras} cameras from {calib_path}')
    return cameras, poses_c2w


ENDOSCOPE_TO_CAM = {1: 1, 2: 2, 3: 3, 4: 4}


def get_camera_index(filename):

    match = re.search(r'_(\d+)\.\w+$', filename)
    if match:
        endoscope_id = int(match.group(1))
        return ENDOSCOPE_TO_CAM.get(endoscope_id, endoscope_id)
    return None


def collect_images(folder):
    exts = ('.jpg', '.jpeg', '.png')
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )
    return files


def undistort_images(image_files, cameras):
    tmpdir = tempfile.mkdtemp(prefix='pow3r_undist_')
    undistorted_files = []
    updated_K = {}
    for filepath in image_files:
        filename = os.path.basename(filepath)
        cam_idx = get_camera_index(filename)
        if cam_idx is None:
            print(f'  Warning: cannot determine camera for {filename}, skipping undistortion')
            undistorted_files.append(filepath)
            continue

        cam_name = f'cam{cam_idx}'
        if cam_name not in cameras:
            print(f'  Warning: {cam_name} not found in calibration, skipping undistortion')
            undistorted_files.append(filepath)
            continue

        img = cv2.imread(filepath)

        K = cameras[cam_name]['K']
        dist = cameras[cam_name]['dist']
        h, w = img.shape[:2]

        newK, roi = cv2.getOptimalNewCameraMatrix(
            K, dist, (w, h), alpha=0, newImgSize=(w, h)
        )

        undist = cv2.undistort(img, K, dist, None, newK)     

        h2, w2 = undist.shape[:2]

        cameras[cam_name]['K'] = newK
        cameras[cam_name]['image_size'] = (w2, h2)
        cameras[cam_name]['dist'] = np.zeros_like(dist)

        out_path = os.path.join(tmpdir, filename)
        cv2.imwrite(out_path, undist)
        undistorted_files.append(out_path)

        updated_K[cam_name] = newK
        cameras[cam_name]['K'] = newK
        print(f'  Undistorted {filename} using {cam_name}')

    return undistorted_files, tmpdir, cameras


def compute_intrinsics_after_resize(K_orig, orig_W, orig_H, target_size=512):
    scale = target_size / max(orig_W, orig_H)
    W = round(orig_W * scale)
    H = round(orig_H * scale)

    K = K_orig.astype(np.float64).copy()
    K[0, :] *= scale
    K[1, :] *= scale

    cx_img, cy_img = W // 2, H // 2
    halfw = ((2 * cx_img) // 16) * 8
    halfh = ((2 * cy_img) // 16) * 8

    crop_x = cx_img - halfw
    crop_y = cy_img - halfh
    K[0, 2] -= crop_x
    K[1, 2] -= crop_y

    return K


def build_calibration_for_images(image_files, cameras, poses_c2w, image_size=512):
    K_list = []
    pose_list = []
    camera_order = []

    for filepath in image_files:
        filename = os.path.basename(filepath)
        cam_idx = get_camera_index(filename)
        cam_name = f'cam{cam_idx}' if cam_idx is not None else None

        if cam_name is None or cam_name not in cameras:
            print(f'  Warning: cannot map {filename} to calibration camera')
            return None, None, None

        cam = cameras[cam_name]
        orig_W, orig_H = cam['image_size']
        K_scaled = compute_intrinsics_after_resize(cam['K'], orig_W, orig_H, image_size)
        K_list.append(K_scaled)

        if cam_name in poses_c2w:
            pose_list.append(poses_c2w[cam_name])
        else:
            pose_list.append(None)

        camera_order.append(cam_name)

    has_all_poses = all(p is not None for p in pose_list)
    if not has_all_poses:
        pose_list = None

    return K_list, pose_list, camera_order


def convert_scene_to_glb(outdir,
                         imgs,
                         focals,
                         cams2world,
                         cam_size=0.05,
                         as_pointcloud=False,
                         all_points=None,
                         all_colors=None):

    scene = trimesh.Scene()

    
    # pct = trimesh.PointCloud(
    #     all_points.reshape(-1, 3),
    #     colors=all_colors.reshape(-1, 3)
    # )
    # scene.add_geometry(pct)
    pct = trimesh.PointCloud(
        all_points,
        colors=all_colors
    )
    scene.add_geometry(pct)

    for i, pose_c2w in enumerate(cams2world):
        camera_edge_color = CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(
            scene,
            pose_c2w,
            camera_edge_color,
            imgs[i],
            float(np.mean(focals[i])),
            imsize=imgs[i].shape[1::-1],
            screen_width=float(cam_size)
        )

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, "scene.glb")

    print("Scene geometries:", scene.geometry.keys())
    print("Number of geometries:", len(scene.geometry))

    scene.export(outfile)

    return outfile


def get_3D_model_from_scene(outdir, scene, min_conf_thr=3, as_pointcloud=False,
                            mask_sky=False, clean_depth=False, cam_size=0.05,
                            image_dir=None):
    if scene is None:
        return None

    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    rgbimg = scene.imgs
    focals = scene.get_focals().detach().cpu().numpy()
    cams2world = scene.get_im_poses().detach().cpu().numpy()


    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))

    pts3d_views = to_numpy(scene.get_pts3d())
    msk_views = to_numpy(scene.get_masks())
    

    n_cams = len(scene.get_im_poses())

    imgs_views = to_numpy(scene.imgs[:n_cams])
    pts3d_views = to_numpy(scene.get_pts3d()[:n_cams])
    msk_views = to_numpy(scene.get_masks()[:n_cams])


    all_points = np.concatenate(
        [pts3d_views[i][msk_views[i]] for i in range(len(pts3d_views))],
        axis=0
    )

    all_colors = np.concatenate(
        [imgs_views[i][msk_views[i]] for i in range(len(imgs_views))],
        axis=0
    )

    # =========================================================
    # 🔹 SEGMENTATION (Luca-style integration)
    # =========================================================
    if image_dir is not None:
        seg_masks = load_segmentation_masks(
            os.path.join(image_dir, "segmentation")
        )

        if seg_masks is not None:
            print(">> Applying segmentation masks")

            for i in range(len(msk_views)):
                sm = seg_masks[i]

                sm = cv2.resize(
                    sm.astype(np.uint8),
                    (pts3d_views[i].shape[1], pts3d_views[i].shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0

                msk_views[i] &= sm
    # -------------------------------------------------
    # FINAL MERGE AFTER SEGMENTATION
    # -------------------------------------------------

    all_points_list = []
    all_colors_list = []

    for i in range(len(pts3d_views)):

        valid_mask = msk_views[i]

        pts = pts3d_views[i][valid_mask]
        cols = imgs_views[i][valid_mask]

        if pts.shape[0] > 0:
            all_points_list.append(pts)
            all_colors_list.append(cols)

    if len(all_points_list) == 0:
        print("⚠ No valid 3D points after masking!")
        all_points = np.zeros((0,3))
        all_colors = np.zeros((0,3))
    else:
        all_points = np.vstack(all_points_list)
        all_colors = np.vstack(all_colors_list)

    print("Merged total points:", all_points.shape[0])


    # -------------------------------------------------
    # Export
    # -------------------------------------------------
    return convert_scene_to_glb(
        outdir=outdir,
        imgs=imgs_views,
        focals=focals,
        cams2world=cams2world,
        cam_size=cam_size,
        as_pointcloud=as_pointcloud,
        all_points=all_points,
        all_colors=all_colors
    )


def merge_views(view_list):
    merged = {}

    for key in view_list[0].keys():
        values = [v[key] for v in view_list]

        if key == "instance":
            merged[key] = sum(values, [])
        elif torch.is_tensor(values[0]):
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list):
            merged[key] = sum(values, [])
        else:
            merged[key] = values

    return merged

def merge_preds(pred_list):
    merged = {}

    for key in pred_list[0].keys():
        values = [p[key] for p in pred_list]

        if torch.is_tensor(values[0]):
            merged[key] = torch.cat(values, dim=0)
        else:
            merged[key] = values

    return merged

def print_dict_structure(d, prefix=""):
    if not isinstance(d, dict):
        print(f"{prefix} (NOT A DICT): {type(d)}")
        return

    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key

        # ---- Dict ----
        if isinstance(value, dict):
            print(f"{path}: DICT (keys={list(value.keys())})")
            print_dict_structure(value, path)

        # ---- Tensor ----
        elif isinstance(value, torch.Tensor):
            print(
                f"{path}: TENSOR "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, "
                f"device={value.device}, "
                f"requires_grad={value.requires_grad}"
            )

        # ---- NumPy ----
        elif isinstance(value, np.ndarray):
            print(
                f"{path}: NUMPY "
                f"shape={value.shape}, "
                f"dtype={value.dtype}"
            )

        # ---- List ----
        elif isinstance(value, list):
            print(f"{path}: LIST len={len(value)}")
            if len(value) > 0:
                print(f"{path}[0]: type={type(value[0])}")

        # ---- Tuple ----
        elif isinstance(value, tuple):
            print(f"{path}: TUPLE len={len(value)}")

        # ---- None ----
        elif value is None:
            print(f"{path}: None")

        # ---- Anything else ----
        else:
            print(f"{path}: type={type(value)}")

# def mat4_to_quat_pose(T):
#     R = T[:3, :3]
#     t = T[:3, 3]

#     # Rotation → Quaternion
#     qw = torch.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
#     qx = (R[2,1] - R[1,2]) / (4*qw)
#     qy = (R[0,2] - R[2,0]) / (4*qw)
#     qz = (R[1,0] - R[0,1]) / (4*qw)

#     return torch.cat([t, torch.tensor([qx, qy, qz, qw])])

# def debug_view(name, view):
#     print(f"\n===== {name} =====")
#     for k, v in view.items():
#         if isinstance(v, torch.Tensor):
#             print(f"{k}: TENSOR shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
#         elif isinstance(v, list):
#             print(f"{k}: LIST len={len(v)}, type[0]={type(v[0]) if len(v)>0 else None}")
#         else:
#             print(f"{k}: {type(v)}")


# def debug_pred(name, pred):
#     print(f"\n===== {name} =====")
#     for k, v in pred.items():
#         if isinstance(v, torch.Tensor):
#             print(f"{k}: TENSOR shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
#         else:
#             print(f"{k}: {type(v)}")

def reconstruct_scene(model, device, image_dir, image_size=512, schedule='linear',
                      niter=300, min_conf_thr=3, as_pointcloud=False,
                      mask_sky=False, clean_depth=True, cam_size=0.05,
                      scenegraph_type='complete', winsize=1, refid=0,
                      outdir='output', calibration=None):

    image_files = collect_images(image_dir)
    if len(image_files) < 2:
        raise RuntimeError(f'Need at least 2 images, found {len(image_files)} in {image_dir}')

    calib_data = None
    tmpdir = None
    if calibration is not None:
        cameras, poses_c2w = load_calibration(calibration)

        print('>> Undistorting images...')
        image_files, tmpdir, cameras = undistort_images(image_files, cameras)

        K_list, pose_list, cam_order = build_calibration_for_images(
            image_files, cameras, poses_c2w, image_size)
        if K_list is not None:
            calib_data = {'K_list': K_list, 'pose_list': pose_list, 'cam_order': cam_order}
            print(f'>> Camera assignment: {cam_order}')

    filelist = image_files
    imgs = load_images(image_files, size=image_size, verbose=True)

    for i, img_entry in enumerate(imgs):
        img_entry["path"] = filelist[i]

    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    print(f'>> Loaded {len(imgs)} images')

    if tmpdir is not None:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    print(f'>> Created {len(pairs)} image pairs (symmetrized)')

    # ------------------------------------------------------------
    # ✅ Inference_with_info für alle Paare
    # ------------------------------------------------------------
    all_view1 = []
    all_view2 = []
    all_pred1 = []
    all_pred2 = []

    image_id_map = {}
    views = []

    # -------- Build stable image ID mapping --------
    for idx, img in enumerate(imgs):
        filename = os.path.basename(img["path"])
        image_id_map[filename] = idx

    # -------- Build views --------
    for img in imgs:
        img_path = img["path"]
        filename = os.path.basename(img_path)

        instance_id = image_id_map[filename]

        view = build_view(
            img,
            img_path,
            cameras,
            poses_c2w,
            instance_id
        )

        view = todevice(view, device)
        views.append(view)

    all_view1 = []
    all_view2 = []
    all_pred1 = []
    all_pred2 = []

    all_points_pred1 = []
    all_points_pred2 = []
    subsample = 20

    for pair in pairs:
###################################################
    # for pair_idx, pair in enumerate(pairs):
    #     if pair_idx >= 2:
    #         break
###################################################

        img1_entry, img2_entry = pair

        img1_filename = os.path.basename(img1_entry["path"])
        img2_filename = os.path.basename(img2_entry["path"])

        img1_id = image_id_map[img1_filename]
        img2_id = image_id_map[img2_filename]

        view1 = views[img1_id]
        view2 = views[img2_id]

        # debug_view("VIEW1", view1)
        # debug_view("VIEW2", view2)

        # print("\n===== EXTRAS =====")
        # print("K1:", view1['camera_intrinsics'].shape)
        # print("K2:", view2['camera_intrinsics'].shape)
        # print("cam1:", view1['camera_pose'].shape)
        # print("cam2:", view2['camera_pose'].shape)


        with torch.no_grad():
            pred1, pred2 = model.inference_with_info(
                view1, view2,
                K1=view1['camera_intrinsics'],
                K2=view2['camera_intrinsics'],
                cam1=view1['camera_pose'],
                cam2=view2['camera_pose']
            )

            # debug_pred("PRED1", pred1)
            # debug_pred("PRED2", pred2)


            pts = pred1["pts3d"][0].reshape(-1,3).cpu().numpy()
            print("Pred1: Pair raw pts mean:", pts.mean(0))
            pts2 = pred2["pts3d_in_other_view"][0].reshape(-1,3).detach().cpu().numpy()
            print("Pred2: Pair raw pts mean:", pts2.mean(0))

        # -------- pred1 --------
        pts1 = pred1["pts3d"][0]                  
        pts1 = pts1[::subsample, ::subsample, :]  
        pts1 = pts1.reshape(-1, 3).cpu().numpy()
        all_points_pred1.append(pts1)

        # -------- pred2 --------
        pts2 = pred2["pts3d_in_other_view"][0]
        pts2 = pts2[::subsample, ::subsample, :]
        pts2 = pts2.reshape(-1, 3).cpu().numpy()
        all_points_pred2.append(pts2)


        all_view1.append(view1)
        all_view2.append(view2)
        all_pred1.append(pred1)
        all_pred2.append(pred2)


    ## pointcloud einzelm ausgeben lassen

    # pair_id = 2

    # pts1 = all_points_pred1[pair_id]
    # pts2 = all_points_pred2[pair_id]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(pts1[:,0], pts1[:,1], pts1[:,2], s=1)
    # ax.scatter(pts2[:,0], pts2[:,1], pts2[:,2], s=1)

    # plt.show()

    # import trimesh
    

    # debug_points = np.vstack([pts1, pts2])

    # # einfache Farbe (weiß)
    # debug_colors = np.ones((debug_points.shape[0], 3)) * 255

    # debug_pc = trimesh.PointCloud(debug_points, colors=debug_colors)

    # debug_file = os.path.join(outdir, f"debug_pair_{pair_id}.glb")
    # debug_pc.export(debug_file)

    # print("Saved debug cloud:", debug_file)

    # subprocess.run([
    #     sys.executable,
    #     r"C:\Users\lucag\Luca_Grab\Studium\Master\Masterarbeit\3D Reconstruction and Grasp Planning\work\pointcloud_vergleich\visualize_glb_p.py",
    #     debug_file,
    #     outdir,
    #     str(0)
    # ], check=True)

    all_points_pred1 = np.concatenate(all_points_pred1, axis=0)
    all_points_pred2 = np.concatenate(all_points_pred2, axis=0)

    print("Total pred1 points:", all_points_pred1.shape)
    print("Total pred2 points:", all_points_pred2.shape)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(
    #     all_points_pred1[:,0],
    #     all_points_pred1[:,1],
    #     all_points_pred1[:,2],
    #     s=1
    # )

    # ax.scatter(
    #     all_points_pred2[:,0],
    #     all_points_pred2[:,1],
    #     all_points_pred2[:,2],
    #     s=1
    # )

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

    # plt.show()

    

    merged_view1 = merge_views(all_view1)
    merged_view2 = merge_views(all_view2)
    merged_pred1 = merge_preds(all_pred1)

    pts = merged_pred1["pts3d"].reshape(-1,3).cpu().numpy()
    print("Pred1: Merged raw cloud:", pts.shape)
    

    merged_pred2 = merge_preds(all_pred2)

    pts2 = merged_pred2["pts3d_in_other_view"].reshape(-1,3).detach().cpu().numpy()
    print("Pred2: Merged raw cloud:", pts2.shape)

    merged_pred1 = {
        "pts3d": merged_pred1["pts3d"],
        "conf": merged_pred1["conf"],
    }

######################################
    for v in [merged_view1, merged_view2]:
        v.pop("camera_intrinsics", None)
        v.pop("camera_pose", None)

    # --- normalize multiview pred2 to DUSt3R format ---
    merged_pred2.pop("pts3d2", None)
    merged_pred2.pop("conf2", None)
######################################

    dust3r_output = {
        "view1": merged_view1,
        "view2": merged_view2,
        "pred1": merged_pred1,
        "pred2": merged_pred2,
        "loss": None,
    }

    # pairs = get_pair_indices(dust3r_output)



    print("\n========== STRUCTURE DEBUG ==========")
    print_dict_structure(dust3r_output)
    print("=====================================\n")


    # ------------------------------------------------------------
    # Global Alignment
    # ------------------------------------------------------------

    mode = (
        GlobalAlignerMode.PointCloudOptimizer
        if len(imgs) > 2
        else GlobalAlignerMode.PairViewer
    )

    print(f'>> Global alignment mode: {mode.value}')

    view1 = dust3r_output["view1"]
    view2 = dust3r_output["view2"]

    # print("view1 idx DAVOR:", view1["idx"])
    # print("view2 idx DAVOR:", view2["idx"])

    unique = sorted(set(view1["idx"]) | set(view2["idx"]))

    print("OLD indices:", unique)

    mapping = {old: new for new, old in enumerate(unique)}

    print("IDX MAPPING:", mapping)

    view1["idx"] = [mapping[i] for i in view1["idx"]]
    view2["idx"] = [mapping[i] for i in view2["idx"]]

    print("NEW indices:", sorted(set(view1["idx"]) | set(view2["idx"])))

    print("view1 idx DANACH:", view1["idx"])
    print("view2 idx DANACH:", view2["idx"])


    scene = global_aligner(
        dust3r_output,
        device=device,
        mode=mode,
        verbose=True
    )

    # --- Get initial point cloud ---
    pts3d_list = scene.get_pts3d()
    pts3d = torch.cat(
        [p.reshape(-1, 3) for p in pts3d_list],
        dim=0
    ).detach().cpu().numpy()

    print("After MST alignment:", pts3d.shape)

    # --- Debug camera poses ---
    poses = scene.get_im_poses().detach().cpu().numpy()

    print("Camera centers:")
    for i, T in enumerate(poses):
        print(i, T[:3, 3])

    # --- Debug number of cameras ---
    print("Number of optimized cameras:", len(scene.get_im_poses()))

    print("\n===== AFTER GLOBAL ALIGNER INIT =====")

    print("Initial focals:")
    print(scene.im_focals.detach().cpu().numpy())

    print("\nInitial principal points:")
    print(scene.im_pp.detach().cpu().numpy())

    print("\nInitial camera poses:")
    poses = scene.get_im_poses().detach().cpu().numpy()
    for i, p in enumerate(poses):
        print(f"Pose {i}:")
        print(p)

    print("=====================================\n")

    pts3d_list = scene.get_pts3d()

    pts3d = torch.cat(
        [p.reshape(-1,3) for p in pts3d_list],
        dim=0
    ).detach().cpu().numpy()
    print("After MST alignment:", pts3d.shape)

    poses = scene.get_im_poses().detach().cpu().numpy()
    print("Camera centers:")
    for i, T in enumerate(poses):
        print(i, T[:3, 3])

    #debug
    print("Number of optimized cameras:", len(scene.get_im_poses()))

    # ------------------------------------------------------------
    # Kalibrierungs-Preset
    # ------------------------------------------------------------

    # --- Freeze intrinsics ---
    if calib_data is not None:
        K_list = calib_data['K_list']
        pose_list = calib_data['pose_list']
        n_cameras = len(set(calib_data['cam_order']))
        n_images = len(imgs)

        known_focals = [float(np.mean([K[0, 0], K[1, 1]])) for K in K_list]

        known_pp = [K[:2, 2] for K in K_list]

        scene.im_pp.requires_grad_(True)
        scene.im_focals.requires_grad_(True)
        scene.im_poses.requires_grad_(True)

        scene.preset_principal_point(known_pp)

        # --- Set focals ---
        scene.preset_focal(known_focals)
        print(f'>> Presetting {n_images} focal lengths and principal points from calibration')

        # # --- Set principal points ---
        # for i, pp in enumerate(known_pp):
        #     H, W = scene.imshapes[i]
        #     scene.im_pp[i].data[:] = (
        #         torch.tensor(pp, dtype=torch.float32)
        #         - torch.tensor([W / 2, H / 2])
        #     )
        #     if scene.verbose:
        #         print(f' (setting principal point #{i} = {pp})')

        # --- Freeze intrinsics ---
        scene.im_focals.requires_grad_(False)
        scene.im_pp.requires_grad_(False)

        # ----------------------------------------------------------
        # --- Set poses ---
        # ----------------------------------------------------------
        if pose_list is not None and n_images == n_cameras:
            known_poses = [
                torch.tensor(p, dtype=torch.float32)
                for p in pose_list
            ]

            print(f'>> Presetting {n_images} camera poses from calibration')
            scene.preset_pose(known_poses)
            scene.im_poses.requires_grad_(False)

        elif pose_list is not None:
            print(
                f'>> Multi-timestep scene ({n_images} images, {n_cameras} cameras): '
                f'intrinsics preset, poses will be estimated'
            )

        print("\n===== BEFORE ALIGNMENT =====")

        print("Focals:")
        print(scene.im_focals.detach().cpu().numpy())

        print("\nPrincipal Points (offset from image center):")
        print(scene.im_pp.detach().cpu().numpy())

        if pose_list is not None:
            print("\nKnown poses from calibration:")
            for i, p in enumerate(pose_list):
                print(f"Pose {i}:")
                print(p)

        print("============================\n")

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        lr = 0.01
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

        print("\n===== AFTER OPTIMIZATION =====")

        print("Optimized focals:")
        print(scene.im_focals.detach().cpu().numpy())

        print("\nOptimized principal points:")
        print(scene.im_pp.detach().cpu().numpy())

        print("\nOptimized poses:")
        poses = scene.get_im_poses().detach().cpu().numpy()
        for i, p in enumerate(poses):
            print(f"Pose {i}:")
            print(p)

        print("==============================\n")

        print(f'>> Global alignment final loss: {loss}')
        pts = pts3d.reshape(-1,3)
        print("Final cloud mean:", pts.mean(0))

    if mode == GlobalAlignerMode.ModularPointCloudOptimizer:

        print(">> Running modular optimizer")

        optimizer = torch.optim.Adam(scene.parameters(), lr=0.01)

        for i in range(niter):
            optimizer.zero_grad()

            loss = scene()   # forward = alignment loss

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"iter {i} loss {loss.item():.6f}")

        print("Final loss:", loss.item())

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

    outfile = get_3D_model_from_scene(outdir, scene, min_conf_thr, as_pointcloud,
                                      mask_sky, clean_depth, cam_size, image_dir=image_dir)

    # ----------------
    # Galerie
    # ----------------
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = plt.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    gallery_imgs = []
    for i in range(len(rgbimg)):
        gallery_imgs.append(rgbimg[i])
        gallery_imgs.append(rgb(depths[i]))
        gallery_imgs.append(rgb(confs[i]))

    # # pair viualisierung
    # pairs = get_pair_indices(dust3r_output)
    # pair_choices = [f"Pair #{i}: views {a}-{b}" for i,(a,b) in enumerate(pairs)]

    # def update_display(selected_pairs, conf_thr, subsample, show_view1, show_view2):

    #     fig = go.Figure()

    #     for pair_choice in selected_pairs:

    #         pair_idx = pair_choices.index(pair_choice)

    #         fig_rgb, _ = build_plots(
    #             dust3r_output,
    #             pair_idx,
    #             conf_thr=conf_thr,
    #             subsample=int(subsample),
    #             show_view1=show_view1,
    #             show_view2=show_view2,
    #             pose_list=None
    #         )

    #         for trace in fig_rgb.data:
    #             fig.add_trace(trace)

    #     fig.update_layout(
    #         scene=dict(aspectmode="data"),
    #         margin=dict(l=0,r=0,b=0,t=30)
    #     )

    #     return fig


    # with gr.Blocks(title="Pow3R Pair Viewer") as demo:

    #     gr.Markdown("# Pow3R Pair Point Cloud Viewer")

    #     with gr.Row():
    #         pair_selector = gr.CheckboxGroup(
    #             choices=pair_choices,
    #             value=[pair_choices[0]],
    #             label="Select pairs"
    #         )

    #         conf_slider = gr.Slider(
    #             0.0, 20.0,
    #             value=1.0,
    #             step=0.5,
    #             label="Confidence threshold"
    #         )

    #         subsample_slider = gr.Slider(
    #             1, 16,
    #             value=4,
    #             step=1,
    #             label="Subsample"
    #         )

    #     with gr.Row():
    #         show_view1 = gr.Checkbox(value=True, label="Show view1")
    #         show_view2 = gr.Checkbox(value=True, label="Show view2")

    #     # with gr.Row():
    #     #     img1_out = gr.Image(label="View1")
    #     #     img2_out = gr.Image(label="View2")

    #     with gr.Row():
    #         plot_rgb = gr.Plot(label="RGB colored")
    #         #plot_rb = gr.Plot(label="Red / Blue")

    #     inputs = [
    #         pair_selector,
    #         conf_slider,
    #         subsample_slider,
    #         show_view1,
    #         show_view2
    #     ]

    #     outputs = [plot_rgb]

    #     pair_selector.change(update_display, inputs=inputs, outputs=outputs)
    #     conf_slider.change(update_display, inputs=inputs, outputs=outputs)
    #     subsample_slider.change(update_display, inputs=inputs, outputs=outputs)
    #     show_view1.change(update_display, inputs=inputs, outputs=outputs)
    #     show_view2.change(update_display, inputs=inputs, outputs=outputs)

    #     demo.load(
    #         fn=lambda: update_display(pair_choices[0],1.0,4,True,True),
    #         outputs=outputs
    #     )

    # demo.launch()

    return scene, outfile, gallery_imgs


def save_gallery(gallery_imgs, outdir, n_views):
    """Save a visual gallery of RGB / depth / confidence per view."""
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(n_views, 3, figsize=(15, 5 * n_views))
    if n_views == 1:
        axes = axes[np.newaxis, :]
    for i in range(n_views):
        for j, title in enumerate(['RGB', 'Depth', 'Confidence']):
            axes[i, j].imshow(gallery_imgs[3 * i + j])
            axes[i, j].set_title(f'View {i} - {title}')
            axes[i, j].axis('off')
    plt.tight_layout()
    gallery_path = os.path.join(outdir, 'gallery.png')
    plt.savefig(gallery_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'>> Saved visual gallery to {gallery_path}')


def parse_args():
    parser = argparse.ArgumentParser('POW3R multi-view 3D reconstruction')

    parser.add_argument('--image_dir', type=str, default='./dataset/scene2',
                        help='Directory containing images to reconstruct')
    parser.add_argument('--ckpt_path', type=str,
                        default='model/Pow3R_ViTLarge_BaseDecoder_512_linear.pth',
                        help='Path to POW3R checkpoint')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Path to calibration JSON (intrinsics, distortion, poses)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='PyTorch device')
    parser.add_argument('--image_size', type=int, default=512, choices=[512, 224],
                        help='Resize images to this resolution')
    parser.add_argument('--outdir', type=str, default='output',
                        help='Output directory for GLB and gallery')

    parser.add_argument('--scenegraph_type', type=str, default='complete',
                        choices=['complete', 'swin', 'oneref'],
                        help='Strategy for creating image pairs')
    parser.add_argument('--winsize', type=int, default=1,
                        help='Window size for swin scene graph')
    parser.add_argument('--refid', type=int, default=0,
                        help='Reference image id for oneref scene graph')

    parser.add_argument('--schedule', type=str, default='linear',
                        choices=['linear', 'cosine'],
                        help='Learning rate schedule for global alignment')
    parser.add_argument('--niter', type=int, default=300,
                        help='Number of iterations for global alignment')
    parser.add_argument('--min_conf_thr', type=float, default=3.0,
                        help='Minimum confidence threshold for filtering')
    parser.add_argument('--cam_size', type=float, default=0.01,
                        help='Camera size in the output scene')

    parser.add_argument('--as_pointcloud', action='store_true',
                        help='Export as point cloud instead of mesh')
    parser.add_argument('--mask_sky', action='store_true',
                        help='Mask sky pixels')
    parser.add_argument('--clean_depth', action='store_true', default=True,
                        help='Clean up depthmaps')

    parser.add_argument('--save_gallery', action='store_true', default=True,
                        help='Save a PNG gallery of RGB/depth/confidence per view')

    return parser.parse_args()

def build_view(img_entry, img_path, cameras, poses_c2w, instance_id):

    cam_id = image_to_cam_id(img_path)

    K = torch.tensor(
        cameras[cam_id]["K"],
        dtype=torch.float32
    )

    if cam_id in poses_c2w:
        pose = torch.tensor(
            poses_c2w[cam_id],
            dtype=torch.float32
        )
    else:
        pose = torch.eye(4, dtype=torch.float32)

    img = img_entry["img"]
    H, W = img.shape[-2:]

    view = {
        "img": img,
        "true_shape": torch.tensor([[H, W]], dtype=torch.int32),
        "instance": [f"cam{instance_id}"],
        "idx": [instance_id],
        "camera_intrinsics": K.unsqueeze(0),
        "camera_pose": pose.unsqueeze(0),
    }

    return view

def image_to_cam_id(img_name: str) -> str:
    name = os.path.basename(img_name)
    match = re.search(r'_([1-4])(?:\D|$)', name)
    if match is None:
        raise ValueError(f"Keine Kamera-ID im Dateinamen gefunden: {name}")
    return f"cam{match.group(1)}"

if __name__ == '__main__':
    args = parse_args()

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model = AsymmetricSliding(
        crop_resolution=(384, 512),
        bootstrap_depth="c2f_both",
        fix_rays="full",
        sparsify_depth=1.1,
    )
    model.load_from_checkpoint(ckpt)
    model = model.to(args.device).eval()

    # 🔥 Direkt Multi-View starten
    scene, outfile, gallery_imgs = reconstruct_scene(
        model=model,
        device=args.device,
        image_dir=args.image_dir,
        image_size=args.image_size,
        schedule=args.schedule,
        niter=args.niter,
        min_conf_thr=args.min_conf_thr,
        as_pointcloud=args.as_pointcloud,
        mask_sky=args.mask_sky,
        clean_depth=args.clean_depth,
        cam_size=args.cam_size,
        scenegraph_type=args.scenegraph_type,
        winsize=args.winsize,
        refid=args.refid,
        outdir=args.outdir,
        calibration=args.calibration,
    )

    print(f'\n>> 3D reconstruction exported to: {outfile}')

    if args.save_gallery:
        n_views = len(gallery_imgs) // 3
        save_gallery(gallery_imgs, args.outdir, n_views)

    print('>> Done!')

    if outfile is not None:
        subprocess.run([
            sys.executable,
            r"C:\Users\lucag\Luca_Grab\Studium\Master\Masterarbeit\3D Reconstruction and Grasp Planning\work\pointcloud_vergleich\visualize_glb_p.py",
            outfile,
            args.outdir,
            str(0) 
        ], check=True)