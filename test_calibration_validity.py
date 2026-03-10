"""
Calibration validity test: load 4 images from a scene, let the user select
one pixel per image (pointing to the same 3D object), then visualize the
rays from each camera in 3D to verify they converge (good calibration).
"""
 
import os
import re
import json
import argparse
import tempfile
import shutil
import numpy as np
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
 
CAM_COLORS = {
    'cam1': 'red',
    'cam2': 'green',
    'cam3': 'blue',
    'cam4': 'magenta',
}
 
 
def load_calibration(calib_path):
    """Load calibration JSON with intrinsics, distortion, and poses."""
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
        # Calibration stores world-to-cam: p_camN = R @ p_cam1 + T
        # Invert to get cam-to-world
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ T
        cam_name = pose_name.split('_to_')[0] if '_to_' in pose_name else pose_name
        poses_c2w[cam_name] = c2w
 
    return cameras, poses_c2w
 
 
def get_camera_index(filename):
    """Extract camera number from filename like '*_endoscope_2.png'."""
    match = re.search(r'_(\d+)\.\w+$', filename)
    if match:
        return int(match.group(1))
    return None
 
 
def collect_scene_images(scene_dir, frame_id=None):
    """
    Collect the 4 camera images for one frame in a scene.
    Returns list of (cam_name, image_path) sorted by cam1, cam2, cam3, cam4.
    """
    exts = ('.jpg', '.jpeg', '.png')
    files = [
        f for f in os.listdir(scene_dir)
        if f.lower().endswith(exts) and 'endoscope' in f.lower()
    ]
 
    # Group by frame id (e.g. 00012 from 00012_endoscope_1.png)
    frames = {}
    for f in files:
        match = re.match(r'(\d+)_endoscope_(\d+)', f, re.IGNORECASE)
        if match:
            fid, cid = match.groups()
            fid, cid = int(fid), int(cid)
            if fid not in frames:
                frames[fid] = {}
            frames[fid][f'cam{cid}'] = os.path.join(scene_dir, f)
 
    if not frames:
        raise FileNotFoundError(f"No images found in {scene_dir}")
 
    # Pick frame
    if frame_id is not None:
        if frame_id not in frames:
            raise FileNotFoundError(f"Frame {frame_id} not found. Available: {sorted(frames.keys())}")
        frame = frames[frame_id]
    else:
        frame_id = min(frames.keys())
        frame = frames[frame_id]
 
    # Ensure we have all 4 cameras
    cam_order = ['cam1', 'cam2', 'cam3', 'cam4']
    result = []
    for c in cam_order:
        if c in frame:
            result.append((c, frame[c]))
        else:
            raise FileNotFoundError(f"Missing {c} in frame {frame_id}")
 
    return result, frame_id
 
 
def undistort_image(img, K, dist):
    """Undistort image using OpenCV."""
    return cv2.undistort(img, K, dist)


def compute_intrinsics_after_resize(K_orig, orig_W, orig_H, target_size=512):
    """Compute the intrinsic matrix after the load_images resize+crop pipeline (same as demo_pow3r_multi_view)."""
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


def _resize_with_dust3r_pipeline(image_list, cam_names, cameras, size=512):
    """
    Use the exact same load_images pipeline as demo_pow3r_multi_view.
    Writes undistorted images to temp dir, calls dust3r load_images, returns
    (img_display_per_cam, cameras_for_rays). Guarantees 100% equivalence.
    """
    import pow3r.tools.path_to_dust3r  # noqa: F401 - sets up dust3r in sys.path
    from PIL import Image
    from dust3r.utils.image import load_images, rgb

    tmpdir = tempfile.mkdtemp(prefix='test_calib_resize_')
    try:
        temp_paths = []
        for cam_name, img_path in image_list:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            cam = cameras[cam_name]
            img_undist = cv2.undistort(img, cam['K'], cam['dist'])
            img_rgb = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)
            out_path = os.path.join(tmpdir, os.path.basename(img_path))
            Image.fromarray(img_rgb).save(out_path)
            temp_paths.append(out_path)

        # load_images expects paths; with list, root='' so paths are used directly
        imgs = load_images(temp_paths, size=size, square_ok=False, verbose=False)

        img_display_per_cam = {}
        cameras_for_rays = {}
        for i, cam_name in enumerate(cam_names):
            img_dict = imgs[i]
            # rgb() returns float [0,1] HWC; convert to BGR uint8 for cv2
            arr = rgb(img_dict['img'])
            if arr.ndim == 4:
                arr = arr[0]
            arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2BGR)
            img_display_per_cam[cam_name] = img_bgr

            H2, W2 = img_dict['true_shape'][0]
            orig_W, orig_H = cameras[cam_name]['image_size']
            K_scaled = compute_intrinsics_after_resize(cameras[cam_name]['K'], orig_W, orig_H, size)
            cameras_for_rays[cam_name] = {
                'K': K_scaled,
                'dist': np.zeros(5, dtype=np.float64),
                'image_size': (W2, H2),
            }

        return img_display_per_cam, cameras_for_rays
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
 
 
def pixel_to_ray(u, v, K, dist):
    """
    Convert pixel (u,v) to ray direction in camera frame.
    Uses undistortPoints to handle distortion.
    """
    pt = np.array([[[u, v]]], dtype=np.float64)
    # Output is normalized camera coordinates (x, y), so ray is [x, y, 1].
    undist_pt = cv2.undistortPoints(pt, K, dist, R=None, P=None)
    x, y = undist_pt[0, 0]
    ray_dir = np.array([x, y, 1.0], dtype=np.float64)
    ray_dir /= np.linalg.norm(ray_dir)
    return ray_dir
 
 
def build_camera_frustum_world(K, image_size, c2w, depth):
    """
    Build camera frustum (origin + 4 image-plane corners) in world frame.
    Uses the linear pinhole model for a clean pyramid shape.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    w, h = image_size

    corners_px = [(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)]
    corners_cam = []
    for u, v in corners_px:
        x = (u - cx) / fx
        y = (v - cy) / fy
        corners_cam.append(np.array([x * depth, y * depth, depth], dtype=np.float64))

    origin_world = c2w[:3, 3]
    R = c2w[:3, :3]
    corners_world = [R @ p + origin_world for p in corners_cam]
    return origin_world, corners_world


def triangulate_rays(origins, directions):
    """
    Find the 3D point closest to all rays (minimize sum of squared distances).
    Uses SVD-based least squares.
    """
    # For each ray: P = O + t*d. Distance from point X to ray: |(X-O) - ((X-O).d)*d|
    # Minimizing sum over rays: sum_i |(X - O_i) - ((X-O_i).d_i)*d_i|^2
    # This is linear in X. Let A_i = I - d_i*d_i^T, b_i = A_i @ O_i
    # Then we minimize sum_i |A_i @ X - b_i|^2 => (sum A_i^T A_i) X = sum A_i^T b_i
    n = len(origins)
    A = np.zeros((3 * n, 3))
    b = np.zeros(3 * n)
    for i in range(n):
        d = directions[i] / np.linalg.norm(directions[i])
        A_i = np.eye(3) - np.outer(d, d)
        A[3*i:3*i+3] = A_i
        b[3*i:3*i+3] = A_i @ origins[i]
    X, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return X
 
 
def select_pixel_interactive(image, window_name, cam_name):
    """Display image and let user click to select a pixel. Returns (u, v)."""
    selected = [None]
 
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected[0] = (x, y)
 
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
 
    display = image.copy()
    if len(display.shape) == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
 
    cv2.putText(display, f"{cam_name}: Click on same object in all views", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
    while selected[0] is None:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(50)
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("User cancelled")
 
    u, v = selected[0]
    cv2.circle(display, (u, v), 5, (0, 255, 0), 2)
    cv2.imshow(window_name, display)
    cv2.waitKey(500)
    cv2.destroyWindow(window_name)
    return u, v
 
 
def plot_rays(origins, directions, cam_names, triangulated_pt, cameras, poses_c2w, ray_length=0.05):
    """Plot rays and triangulated point using Matplotlib."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    frustum_depth = ray_length * 0.15
 
    for i, (orig, d) in enumerate(zip(origins, directions)):
        d_norm = d / np.linalg.norm(d)
        end = orig + ray_length * d_norm
        cam_name = cam_names[i]
        color = CAM_COLORS.get(cam_name, 'gray')
 
        ax.plot([orig[0], end[0]], [orig[1], end[1]], [orig[2], end[2]],
                color=color, linewidth=2, label=cam_name)
        ax.scatter(*orig, color=color, s=30)

        fr_origin, fr_corners = build_camera_frustum_world(
            cameras[cam_name]['K'],
            cameras[cam_name]['image_size'],
            poses_c2w[cam_name],
            frustum_depth
        )
        segments = [
            (fr_origin, fr_corners[0]),
            (fr_origin, fr_corners[1]),
            (fr_origin, fr_corners[2]),
            (fr_origin, fr_corners[3]),
            (fr_corners[0], fr_corners[1]),
            (fr_corners[1], fr_corners[2]),
            (fr_corners[2], fr_corners[3]),
            (fr_corners[3], fr_corners[0]),
        ]
        for p0, p1 in segments:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=color, linewidth=1.2, alpha=0.65)
 
    ax.scatter(*triangulated_pt, color='black', s=100, marker='*', label='Triangulated')
 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Calibration check: rays should converge at the triangulated point')
    plt.tight_layout()
    plt.show()
 
 
def main():
    parser = argparse.ArgumentParser(
        description='Verify calibration quality by selecting the same point in 4 views and visualizing rays in 3D'
    )
    parser.add_argument('--scene', type=str, default='dataset_v2/scene2',
                        help='Path to scene directory')
 
    parser.add_argument('--frame', type=int, default=None,
                        help='Frame ID (e.g. 12 for 00012). Default: first available')
 
    parser.add_argument('--calibration', type=str, default='dataset_v2/calibration.json',
                        help='Path to calibration JSON')
 
    parser.add_argument('--ray_length', type=float, default=0.15,
                        help='Length of rays to display in 3D (meters)')

    parser.add_argument('--resize', action='store_true',
                        help='Resize and crop images like demo_pow3r_multi_view (to verify resize pipeline)')
    parser.add_argument('--resize_size', type=int, default=512,
                        help='Target size for resize (default: 512, same as demo)')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Directory to save undistorted images (default: current dir)')

    args = parser.parse_args()
 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(script_dir, args.scene) if not os.path.isabs(args.scene) else args.scene
    calib_path = os.path.join(script_dir, args.calibration) if not os.path.isabs(args.calibration) else args.calibration
 
    if not os.path.isdir(scene_path):
        raise FileNotFoundError(f"Scene directory not found: {scene_path}")
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
 
    print(f"Loading calibration from {calib_path}")
    cameras, poses_c2w = load_calibration(calib_path)
 
    print(f"Loading images from {scene_path}")
    image_list, frame_id = collect_scene_images(scene_path, args.frame)
    print(f"Using frame {frame_id}")
    if args.resize:
        print(f"Resize mode: ON (target_size={args.resize_size}, same as demo_pow3r_multi_view)")

    # Load and undistort images, optionally resize+crop (uses dust3r load_images when --resize)
    pixels = {}
    cameras_for_rays = {}  # K, dist, image_size per cam (resized or original)
    cam_names_ordered = [c for c, _ in image_list]

    if args.resize:
        # Use exact dust3r load_images pipeline (100% equivalent to demo_pow3r_multi_view)
        img_display_per_cam, cameras_for_rays = _resize_with_dust3r_pipeline(
            image_list, cam_names_ordered, cameras, size=args.resize_size
        )
        for cam_name in cam_names_ordered:
            orig_W, orig_H = cameras[cam_name]['image_size']
            crop_W, crop_H = cameras_for_rays[cam_name]['image_size']
            print(f"  {cam_name}: resized {orig_W}x{orig_H} -> {crop_W}x{crop_H}")
    else:
        img_display_per_cam = {}
        for cam_name, img_path in image_list:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            cam = cameras[cam_name]
            img_display_per_cam[cam_name] = undistort_image(img, cam['K'], cam['dist'])
            cameras_for_rays[cam_name] = cam

    # 1. Print camera position and associated image name
    print("\n>> Camera positions and associated images:")
    for cam_name, img_path in image_list:
        pos = poses_c2w[cam_name][:3, 3]
        img_name = os.path.basename(img_path)
        print(f"   {cam_name}: {img_name} -> position ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})")

    # 2. Save undistorted images as calibration_validity_image_x.png
    outdir = os.path.join(script_dir, args.outdir) if not os.path.isabs(args.outdir) else args.outdir
    os.makedirs(outdir, exist_ok=True)
    for cam_name in cam_names_ordered:
        cam_num = cam_name.replace('cam', '')
        out_path = os.path.join(outdir, f"calibration_validity_image_{cam_num}.png")
        cv2.imwrite(out_path, img_display_per_cam[cam_name])
        print(f"   Saved {out_path}")

    for cam_name in cam_names_ordered:
        print(f"\nSelect a pixel in {cam_name} (click on the same object visible in all 4 views)")
        u, v = select_pixel_interactive(
            img_display_per_cam[cam_name], f"Select point - {cam_name}", cam_name
        )
        pixels[cam_name] = (u, v)
        print(f"  Selected: ({u}, {v})")

    cv2.destroyAllWindows()

    # Compute rays in world frame (cam1 frame)
    origins = []
    directions = []
    cam_names = []

    for cam_name, (u, v) in pixels.items():
        cam = cameras_for_rays[cam_name]
        c2w = poses_c2w[cam_name]

        ray_dir_cam = pixel_to_ray(u, v, cam['K'], cam['dist'])
        ray_origin_world = c2w[:3, 3]
        ray_dir_world = c2w[:3, :3] @ ray_dir_cam
 
        origins.append(ray_origin_world)
        directions.append(ray_dir_world)
        cam_names.append(cam_name)
 
    # Triangulate the 3D point
    triangulated = triangulate_rays(origins, directions)
    print(f"\nTriangulated 3D point: {triangulated}")
 
    # Compute reprojection error (distance from each ray to triangulated point)
    print("\nReprojection error (distance from each ray to triangulated point):")
    for i, (orig, d) in enumerate(zip(origins, directions)):
        d_norm = d / np.linalg.norm(d)
        # Project triangulated point onto ray
        t = np.dot(triangulated - orig, d_norm)
        closest = orig + t * d_norm
        dist = np.linalg.norm(triangulated - closest)
        print(f"  {cam_names[i]}: {dist*1000:.2f} mm")
 
    # Plot (use cameras_for_rays so frustums match the displayed/resized view)
    plot_rays(origins, directions, cam_names, triangulated, cameras_for_rays, poses_c2w, args.ray_length)
 
 
if __name__ == '__main__':
    main()