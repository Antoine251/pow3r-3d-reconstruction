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

import cv2
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as pl
from scipy.spatial.transform import Rotation

import pow3r.tools.path_to_dust3r
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from pow3r.model import Pow3R  # noqa: F401 - needed for eval(ckpt['definition'])


def load_pow3r_model(ckpt_path, device='cuda'):
    """Load the base POW3R model from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_def = ckpt['definition']
    print(f'>> Creating POW3R model = {model_def}')
    model = eval(model_def)
    print(f'>> Loading weights:', model.load_state_dict(ckpt['weights']))
    return model.to(device).eval()


# --------------------------------------------------------
# Calibration utilities
# --------------------------------------------------------

def load_calibration(calib_path):
    """Load calibration JSON containing per-camera intrinsics, distortion, and poses."""
    with open(calib_path) as f:
        calib = json.load(f)

    cameras = {}
    for cam_name, cam_data in calib['cameras'].items():
        cameras[cam_name] = {
            'K': np.array(cam_data['K'], dtype=np.float64),
            'dist': np.array(cam_data['dist'], dtype=np.float64),
            'image_size': tuple(cam_data['image_size']),
        }

    poses_c2w = {}
    for pose_name, pose_data in calib['camera_poses'].items():
        R = np.array(pose_data['R'], dtype=np.float64)
        T = np.array(pose_data['T'], dtype=np.float64)
        # Calibration stores world-to-cam: p_camN = R @ p_cam1 + T
        # Invert to get cam-to-world: position = -R^T @ T, orientation = R^T
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
    """Extract camera number (1-based) from filename like '*_endoscope_2.png' or '*_2.png',
    then remap to calibration camera index via ENDOSCOPE_TO_CAM."""
    match = re.search(r'_(\d+)\.\w+$', filename)
    if match:
        endoscope_id = int(match.group(1))
        return ENDOSCOPE_TO_CAM.get(endoscope_id, endoscope_id)
    return None


def collect_images(folder):
    """Collect image files sorted by name, excluding non-image files."""
    exts = ('.jpg', '.jpeg', '.png')
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )
    return files


def undistort_images(image_files, cameras):
    """Undistort images using per-camera calibration. Returns temp file paths."""
    tmpdir = tempfile.mkdtemp(prefix='pow3r_undist_')
    undistorted_files = []
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
        undist = cv2.undistort(img, K, dist)
        out_path = os.path.join(tmpdir, filename)
        cv2.imwrite(out_path, undist)
        undistorted_files.append(out_path)
        print(f'  Undistorted {filename} using {cam_name}')

    return undistorted_files, tmpdir


def compute_intrinsics_after_resize(K_orig, orig_W, orig_H, target_size=512):
    """Compute the intrinsic matrix after the load_images resize+crop pipeline."""
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
    """Build per-image intrinsic matrices (scaled) and cam-to-world poses.

    Returns:
        K_list: list of 3x3 numpy arrays (one per image)
        pose_list: list of 4x4 numpy arrays (one per image), or None if mapping fails
        camera_order: list of camera names for each image
    """
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


# --------------------------------------------------------
# GLB export and scene extraction
# --------------------------------------------------------

def convert_scene_to_glb(outdir, imgs, pts3d, mask, focals, cams2world,
                         cam_size=0.05, as_pointcloud=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    for i, pose_c2w in enumerate(cams2world):
        camera_edge_color = CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, 'scene.glb')
    print(f'(exporting 3D scene to {outfile})')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, scene, min_conf_thr=3, as_pointcloud=False,
                            mask_sky=False, clean_depth=False, cam_size=0.05):
    if scene is None:
        return None
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return convert_scene_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world,
                                as_pointcloud=as_pointcloud, cam_size=cam_size)


# --------------------------------------------------------
# Main reconstruction pipeline
# --------------------------------------------------------

def reconstruct_scene(model, device, image_dir, image_size=512, schedule='linear',
                      niter=300, min_conf_thr=3, as_pointcloud=False,
                      mask_sky=False, clean_depth=True, cam_size=0.05,
                      scenegraph_type='complete', winsize=1, refid=0,
                      outdir='output', calibration=None):
    """
    Full multi-view reconstruction pipeline:
    1. Load images (optionally undistort with calibration)
    2. Create pairs according to the scene graph strategy
    3. Run POW3R pairwise inference
    4. Globally align all views (optionally preset known intrinsics/poses)
    5. Export GLB
    """
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

    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    print(f'>> Created {len(pairs)} image pairs (symmetrized)')

    output = inference(pairs, model, device, batch_size=1, verbose=True)

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
            scene.im_pp[i].data[:] = torch.tensor(pp, dtype=torch.float32) - torch.tensor([W / 2, H / 2])
            if scene.verbose:
                print(f' (setting principal point #{i} = {pp})')

        if pose_list is not None and n_images == n_cameras:
            known_poses = [torch.tensor(p, dtype=torch.float32) for p in pose_list]
            print(f'>> Presetting {n_images} camera poses from calibration')
            scene.preset_pose(known_poses)
        elif pose_list is not None:
            print(f'>> Multi-timestep scene ({n_images} images, {n_cameras} cameras): '
                  f'intrinsics preset, poses will be estimated')

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        lr = 0.01
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
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

    outfile = get_3D_model_from_scene(outdir, scene, min_conf_thr, as_pointcloud,
                                      mask_sky, clean_depth, cam_size)

    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    gallery_imgs = []
    for i in range(len(rgbimg)):
        gallery_imgs.append(rgbimg[i])
        gallery_imgs.append(rgb(depths[i]))
        gallery_imgs.append(rgb(confs[i]))

    return scene, outfile, gallery_imgs


def save_gallery(gallery_imgs, outdir, n_views):
    """Save a visual gallery of RGB / depth / confidence per view."""
    os.makedirs(outdir, exist_ok=True)
    fig, axes = pl.subplots(n_views, 3, figsize=(15, 5 * n_views))
    if n_views == 1:
        axes = axes[np.newaxis, :]
    for i in range(n_views):
        for j, title in enumerate(['RGB', 'Depth', 'Confidence']):
            axes[i, j].imshow(gallery_imgs[3 * i + j])
            axes[i, j].set_title(f'View {i} - {title}')
            axes[i, j].axis('off')
    pl.tight_layout()
    gallery_path = os.path.join(outdir, 'gallery.png')
    pl.savefig(gallery_path, dpi=150, bbox_inches='tight')
    pl.close()
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
    parser.add_argument('--device', type=str, default='cuda',
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


if __name__ == '__main__':
    args = parse_args()

    model = load_pow3r_model(args.ckpt_path, device=args.device)

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
