# Textual Inversion Image Generator

This is a Gradio-based web application that generates images using both Stable Diffusion 1.5 and Stable Diffusion XL with Textual Inversion. The application includes two tabs:
- SD 1.5 tab: Uses the "cat-toy" concept from the SD Concepts Library
- SDXL tab: Uses the "unaestheticXL" embedding as a negative prompt for improved image quality

## Requirements

- Python 3.8 or higher
- 16GB+ RAM recommended (especially for SDXL)
- CPU with 4+ cores recommended
- 10GB+ free disk space for model downloads

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://127.0.0.1:7860)

3. Choose between SD 1.5 and SDXL tabs:

   **SD 1.5 Tab:**
   - Enter your prompt (use `<cat-toy>` to reference the learned concept)
   - Adjust the number of images to generate (1-4)
   - Modify the guidance scale (1.0-20.0)
   - Change the number of inference steps (20-50)
   - Click "Generate Images" to create your images

   **SDXL Tab:**
   - Enter your prompt (default: "a woman standing in front of a mountain")
   - Adjust the number of images to generate (1-4)
   - Modify the guidance scale (1.0-20.0)
   - Change the number of inference steps (20-50)
   - Click "Generate Images" to create your images
   - Note: The unaestheticXL embedding is automatically used as a negative prompt

## Features

- Interactive web interface with tabbed navigation
- Support for both SD 1.5 and SDXL models
- Customizable generation parameters
- Grid display of generated images
- Support for Textual Inversion concepts
- CPU-optimized settings

## Performance Notes

- The application is configured to run on CPU
- Image generation will be significantly slower compared to GPU versions
- SDXL generation will be slower than SD 1.5 due to the larger model size
- Default settings are optimized for CPU performance:
  - Maximum 4 images per generation
  - Maximum 50 inference steps
  - Using float32 precision for better CPU compatibility
- Recommended settings for faster generation:
  - Use 20-30 inference steps
  - Generate 1-2 images at a time
  - Be patient as each generation might take several minutes

## Note

The first time you run the application, it will download:
- Stable Diffusion 1.5 model
- SDXL model
- Textual Inversion embeddings (cat-toy and unaestheticXL)

This initial download may take some time depending on your internet connection. The models will be cached locally for faster subsequent runs. 