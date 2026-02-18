# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# Multi-view 3D reconstruction using DUSt3R.
# Runs DUSt3R inference on all image pairs, then performs global alignment
# to produce a coherent 3D reconstruction from multiple views.

import os
import argparse
import copy
import builtins
import datetime

import cv2
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as pl
from scipy.spatial.transform import Rotation

import pow3r.tools.path_to_dust3r
from dust3r.model import AsymmetricCroCo3DStereo  # noqa: F401 - needed for eval()
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


def set_print_with_timestamp(time_format="%Y-%m-%d %H:%M:%S"):
    builtin_print = builtins.print

    def print_with_timestamp(*args, **kwargs):
        now = datetime.datetime.now().strftime(time_format)
        builtin_print(f'[{now}]', end=' ')
        builtin_print(*args, **kwargs)

    builtins.print = print_with_timestamp


def collect_images(folder):
    exts = ('.jpg', '.jpeg', '.png')
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    )
    if len(files) < 2:
        raise RuntimeError(f'At least 2 images are required, found {len(files)} in {folder}')
    return files


def load_segmentation_masks(folder_path):
    """Load binary masks from image_dir/segmentation/ (files with 'mask' in name)."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    mask_files = sorted([
        f for f in os.listdir(folder_path)
        if 'mask' in f.lower() and f.lower().endswith(exts)
    ])

    if not mask_files:
        print(f'Warning: No mask files found in {folder_path}')
        return None

    binary_masks = []
    for filename in mask_files:
        mask_img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f'Skipping {filename}: could not read')
            continue
        _, binary_mask = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)
        binary_masks.append(binary_mask)

    print(f'Loaded {len(binary_masks)} segmentation masks.')
    return binary_masks


def export_scene_glb(outdir, imgs, pts3d, mask, focals, cams2world,
                     seg_masks=None, cam_size=0.05, as_pointcloud=False):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    if seg_masks is not None:
        seg_masks = to_numpy(seg_masks)
        for i in range(len(mask)):
            h, w = mask[i].shape
            s_mask = cv2.resize(seg_masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
            mask[i] = s_mask > 0

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
    print(f'Exported: {outfile}')
    scene.export(file_obj=outfile)
    return outfile


def reconstruct_scene(model, device, image_dir, image_size=512,
                      schedule='linear', niter=300, min_conf_thr=3.0,
                      as_pointcloud=False, cam_size=0.05,
                      scenegraph_type='complete', winsize=1, refid=0,
                      outdir='output_dust3r'):
    image_files = collect_images(image_dir)

    try:
        square_ok = model.square_ok
    except AttributeError:
        square_ok = False

    imgs = load_images(
        image_files,
        size=image_size,
        verbose=True,
        square_ok=square_ok,
    )
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    print(f'>> Loaded {len(imgs)} images')

    seg_mask_dir = os.path.join(image_dir, 'segmentation')
    seg_masks = None
    if os.path.isdir(seg_mask_dir):
        seg_masks = load_segmentation_masks(seg_mask_dir)

    if scenegraph_type == 'swin':
        scenegraph_type = scenegraph_type + '-' + str(winsize)
    elif scenegraph_type == 'oneref':
        scenegraph_type = scenegraph_type + '-' + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    print(f'>> Created {len(pairs)} image pairs (symmetrized)')

    print('Running DUSt3R inference...')
    output = inference(pairs, model, device, batch_size=1, verbose=True)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    print(f'>> Global alignment mode: {mode.value}')

    print('Running global alignment...')
    scene = global_aligner(output, device=device, mode=mode, verbose=True)

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        lr = 0.01
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
        print(f'>> Global alignment final loss: {loss}')

    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))

    outfile = export_scene_glb(
        outdir=outdir,
        imgs=scene.imgs,
        pts3d=scene.get_pts3d(),
        mask=scene.get_masks(),
        focals=scene.get_focals().cpu(),
        cams2world=scene.get_im_poses().cpu(),
        seg_masks=seg_masks,
        cam_size=cam_size,
        as_pointcloud=as_pointcloud,
    )

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
    parser = argparse.ArgumentParser('DUSt3R multi-view 3D reconstruction')

    parser.add_argument('--image_dir', type=str, default='./dataset/scene2',
                        help='Directory containing images to reconstruct')
    parser.add_argument('--device', type=str, default='cuda',
                        help='PyTorch device')
    parser.add_argument('--image_size', type=int, default=512, choices=[512, 224],
                        help='Resize images to this resolution')
    parser.add_argument('--outdir', type=str, default='output_dust3r',
                        help='Output directory for GLB and gallery')

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--weights', type=str, default=None,
                             help='Path to local DUSt3R checkpoint (.pth file)')
    model_group.add_argument('--model_name', type=str,
                             default='DUSt3R_ViTLarge_BaseDecoder_512_dpt',
                             choices=['DUSt3R_ViTLarge_BaseDecoder_512_dpt',
                                      'DUSt3R_ViTLarge_BaseDecoder_512_linear',
                                      'DUSt3R_ViTLarge_BaseDecoder_224_linear'],
                             help='HuggingFace model name (downloads if needed)')

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
    parser.add_argument('--cam_size', type=float, default=0.05,
                        help='Camera size in the output scene')

    parser.add_argument('--as_pointcloud', action='store_true',
                        help='Export as point cloud instead of mesh')

    parser.add_argument('--save_gallery', action='store_true', default=True,
                        help='Save a PNG gallery of RGB/depth/confidence per view')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_print_with_timestamp()

    if args.weights:
        print(f'Loading model from local weights: {args.weights}')
        model = AsymmetricCroCo3DStereo.from_pretrained(args.weights)
    else:
        print(f'Loading model: {args.model_name}')
        model = AsymmetricCroCo3DStereo.from_pretrained('naver/' + args.model_name)

    model = model.to(args.device).eval()

    scene, outfile, gallery_imgs = reconstruct_scene(
        model=model,
        device=args.device,
        image_dir=args.image_dir,
        image_size=args.image_size,
        schedule=args.schedule,
        niter=args.niter,
        min_conf_thr=args.min_conf_thr,
        as_pointcloud=args.as_pointcloud,
        cam_size=args.cam_size,
        scenegraph_type=args.scenegraph_type,
        winsize=args.winsize,
        refid=args.refid,
        outdir=args.outdir,
    )

    print(f'\n>> 3D reconstruction exported to: {outfile}')

    if args.save_gallery:
        n_views = len(gallery_imgs) // 3
        save_gallery(gallery_imgs, args.outdir, n_views)

    print('>> Done!')
