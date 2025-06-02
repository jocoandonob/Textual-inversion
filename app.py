import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid

def load_model():
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    repo_id_embeds = "sd-concepts-library/cat-toy"
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    
    pipeline.load_textual_inversion(repo_id_embeds)
    return pipeline

def generate_images(prompt, num_images, guidance_scale, num_inference_steps):
    pipeline = load_model()
    
    # Calculate grid dimensions
    num_samples_per_row = 2
    num_rows = (num_images + 1) // 2  # Round up division
    
    all_images = []
    for _ in range(num_rows):
        images = pipeline(
            prompt,
            num_images_per_prompt=min(num_samples_per_row, num_images - len(all_images)),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images
        all_images.extend(images)
    
    grid = make_image_grid(all_images, num_rows, num_samples_per_row)
    return grid

# Create Gradio interface
with gr.Blocks(title="Textual Inversion Image Generator") as demo:
    gr.Markdown("# Textual Inversion Image Generator")
    gr.Markdown("Generate images using Textual Inversion with Stable Diffusion")
    gr.Markdown("⚠️ Running on CPU - Generation will be slower than GPU")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                value="a grafitti in a favela wall with a <cat-toy> on it"
            )
            num_images = gr.Slider(
                minimum=1,
                maximum=4,
                value=2,
                step=1,
                label="Number of Images"
            )
            guidance_scale = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                value=7.5,
                step=0.1,
                label="Guidance Scale"
            )
            num_inference_steps = gr.Slider(
                minimum=20,
                maximum=50,
                value=30,
                step=1,
                label="Number of Inference Steps"
            )
            generate_btn = gr.Button("Generate Images")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Images")
    
    generate_btn.click(
        fn=generate_images,
        inputs=[prompt, num_images, guidance_scale, num_inference_steps],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(share=True) 