# Textual Inversion Image Generator

This is a Gradio-based web application that generates images using Stable Diffusion with Textual Inversion. The application uses the "cat-toy" concept from the SD Concepts Library.

## Requirements

- Python 3.8 or higher
- 8GB+ RAM recommended
- CPU with 4+ cores recommended

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

3. Use the interface to:
   - Enter your prompt (use `<cat-toy>` to reference the learned concept)
   - Adjust the number of images to generate (1-4)
   - Modify the guidance scale (1.0-20.0)
   - Change the number of inference steps (20-50)
   - Click "Generate Images" to create your images

## Features

- Interactive web interface
- Customizable generation parameters
- Grid display of generated images
- Support for Textual Inversion concepts
- CPU-optimized settings

## Performance Notes

- The application is configured to run on CPU
- Image generation will be slower compared to GPU versions
- Default settings are optimized for CPU performance:
  - Maximum 4 images per generation
  - Maximum 50 inference steps
  - Using float32 precision for better CPU compatibility

## Note

The first time you run the application, it will download the Stable Diffusion model and the Textual Inversion embeddings, which may take some time depending on your internet connection. 