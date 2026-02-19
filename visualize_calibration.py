import json
import argparse
import numpy as np
import plotly.graph_objects as go
import gradio as gr


CAM_COLORS = {
    'cam1': 'red',
    'cam2': 'green',
    'cam3': 'blue',
    'cam4': 'magenta',
}


# ------------------------------
# Loading
# ------------------------------

def load_calibration(path):
    with open(path) as f:
        calib = json.load(f)

    cameras = {}
    for cam_name, cam_data in calib['cameras'].items():
        cameras[cam_name] = {
            'K': np.array(cam_data['K'], dtype=np.float64),
            'image_size': cam_data['image_size'],
        }

    poses_c2w = {}
    for pose_name, pose_data in calib['camera_poses'].items():
        R = np.array(pose_data['R'], dtype=np.float64)
        T = np.array(pose_data['T'], dtype=np.float64)
        # Calibration stores world-to-cam: p_camN = R @ p_cam1 + T
        # Invert to get cam-to-world
        c2w = np.eye(4)
        c2w[:3,:3] = R.T
        c2w[:3,3] = -R.T @ T
        cam_name = pose_name.split('_to_')[0]
        poses_c2w[cam_name] = c2w

    return cameras, poses_c2w


# ------------------------------
# Geometry creation
# ------------------------------

def frustum_points(K, c2w, W, H, depth):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    corners_cam = np.array([
        [0,0,0],
        [(0-cx)/fx*depth, (0-cy)/fy*depth, depth],
        [(W-cx)/fx*depth, (0-cy)/fy*depth, depth],
        [(W-cx)/fx*depth, (H-cy)/fy*depth, depth],
        [(0-cx)/fx*depth, (H-cy)/fy*depth, depth],
    ])

    R = c2w[:3,:3]
    t = c2w[:3,3]
    return (R @ corners_cam.T).T + t


LINES = [(0,1),(0,2),(0,3),(0,4),
         (1,2),(2,3),(3,4),(4,1)]


# ------------------------------
# Plotly Rendering
# ------------------------------

def render_plot(calibration, depth):
    cameras, poses = load_calibration(calibration)
    fig = go.Figure()

    for cam_name in sorted(cameras):
        if cam_name not in poses:
            continue

        cam = cameras[cam_name]
        pts = frustum_points(
            cam['K'],
            poses[cam_name],
            *cam['image_size'],
            depth
        )

        color = CAM_COLORS.get(cam_name, 'gray')

        # draw lines
        for a,b in LINES:
            fig.add_trace(go.Scatter3d(
                x=[pts[a,0], pts[b,0]],
                y=[pts[a,1], pts[b,1]],
                z=[pts[a,2], pts[b,2]],
                mode='lines',
                line=dict(color=color,width=4),
                showlegend=False
            ))

        # draw camera center
        fig.add_trace(go.Scatter3d(
            x=[pts[0,0]],
            y=[pts[0,1]],
            z=[pts[0,2]],
            mode='markers+text',
            marker=dict(size=4,color=color),
            text=[cam_name],
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            aspectmode='data'
        ),
        margin=dict(l=0,r=0,t=30,b=0)
    )

    return fig


# ------------------------------
# Gradio UI
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', default='dataset_v2/calibration.json')
    parser.add_argument('--depth', type=float, default=0.03)
    args = parser.parse_args()

    demo = gr.Interface(
        fn=render_plot,
        inputs=[
            gr.Textbox(value=args.calibration, label="Calibration file"),
            gr.Slider(0.005,0.2,value=args.depth,label="Frustum depth")
        ],
        outputs=gr.Plot(),
        title="Camera Frustum Viewer (No Open3D)"
    )

    demo.launch()
