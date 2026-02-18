import gradio as gr


def launch_viewer(file_path):
    with gr.Blocks() as demo:
        gr.Markdown(f"### Visualizing: {file_path}")
        # Gradio Model3D supports .glb natively with textures/colors
        gr.Model3D(file_path, label="DUSt3R Reconstruction")
        
    demo.launch(share=True)


if __name__ == "__main__":
    launch_viewer("./output/scene.glb")