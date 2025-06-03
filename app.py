import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
from diffusers.utils import make_image_grid
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def load_sd_model():
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    repo_id_embeds = "sd-concepts-library/cat-toy"
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    
    pipeline.load_textual_inversion(repo_id_embeds)
    return pipeline

def load_sdxl_model():
    # Load SDXL pipeline
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32  # Changed from float16 to float32 for CPU
    )
    
    # Load unaestheticXL embeddings
    file = hf_hub_download("dn118/unaestheticXL", filename="unaestheticXLv31.safetensors")
    state_dict = load_file(file)
    
    # Load embeddings into both text encoders
    pipe.load_textual_inversion(
        state_dict["clip_g"],
        token="unaestheticXLv31",
        text_encoder=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer_2
    )
    pipe.load_textual_inversion(
        state_dict["clip_l"],
        token="unaestheticXLv31",
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer
    )
    
    return pipe

def generate_sd_images(prompt, num_images, guidance_scale, num_inference_steps):
    pipeline = load_sd_model()
    
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

def generate_sdxl_images(prompt, num_images, guidance_scale, num_inference_steps):
    pipeline = load_sdxl_model()
    
    # Calculate grid dimensions
    num_samples_per_row = 2
    num_rows = (num_images + 1) // 2  # Round up division
    
    all_images = []
    for _ in range(num_rows):
        images = pipeline(
            prompt,
            negative_prompt="unaestheticXLv31",  # Use unaestheticXL as negative prompt
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
    
    with gr.Tabs():
        with gr.TabItem("Stable Diffusion 1.5"):
            with gr.Row():
                with gr.Column():
                    sd_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        value="a grafitti in a favela wall with a <cat-toy> on it"
                    )
                    sd_num_images = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=2,
                        step=1,
                        label="Number of Images"
                    )
                    sd_guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.1,
                        label="Guidance Scale"
                    )
                    sd_num_inference_steps = gr.Slider(
                        minimum=20,
                        maximum=50,
                        value=30,
                        step=1,
                        label="Number of Inference Steps"
                    )
                    sd_generate_btn = gr.Button("Generate Images")
                
                with gr.Column():
                    sd_output_image = gr.Image(label="Generated Images")
            
            sd_generate_btn.click(
                fn=generate_sd_images,
                inputs=[sd_prompt, sd_num_images, sd_guidance_scale, sd_num_inference_steps],
                outputs=sd_output_image
            )
        
        with gr.TabItem("Stable Diffusion XL"):
            with gr.Row():
                with gr.Column():
                    sdxl_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        value="a woman standing in front of a mountain"
                    )
                    sdxl_num_images = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=2,
                        step=1,
                        label="Number of Images"
                    )
                    sdxl_guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.1,
                        label="Guidance Scale"
                    )
                    sdxl_num_inference_steps = gr.Slider(
                        minimum=20,
                        maximum=50,
                        value=30,
                        step=1,
                        label="Number of Inference Steps"
                    )
                    sdxl_generate_btn = gr.Button("Generate Images")
                
                with gr.Column():
                    sdxl_output_image = gr.Image(label="Generated Images")
            
            sdxl_generate_btn.click(
                fn=generate_sdxl_images,
                inputs=[sdxl_prompt, sdxl_num_images, sdxl_guidance_scale, sdxl_num_inference_steps],
                outputs=sdxl_output_image
            )

if __name__ == "__main__":
    demo.launch(share=True) 