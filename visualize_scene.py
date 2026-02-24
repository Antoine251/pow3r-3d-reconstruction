import argparse
 
import gradio as gr
 
 
def launch_viewer(file_path):
    with gr.Blocks() as demo:
        gr.Markdown(f"### Visualizing: {file_path}")
        # Gradio Model3D supports .glb natively with textures/colors
        gr.Model3D(file_path, label="DUSt3R Reconstruction")
       
    demo.launch(share=True)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a 3D scene (.glb file)")
    parser.add_argument(
        "--scene_path", "-s",
        default="output/scene.glb",
        help="Path to the scene.glb file (default: output/scene.glb)",
    )
    args = parser.parse_args()
    launch_viewer(args.scene_path)