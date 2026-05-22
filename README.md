# Zero-Shot Interactive Trash Sorter

CLIP-powered waste classification using zero-shot learning. Upload any image and define custom categories with text labels — no retraining needed.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Open http://localhost:7860 in your browser.

## Project Structure

```
zero-shot-trash-sorter/
├── engine.py      # TrashClassifier (CLIP model, prediction logic)
├── optimize.py    # INT8 quantization script
├── app.py         # Gradio web interface
├── main.py        # Entry point
├── requirements.txt
└── examples/      # Sample images for testing
```

## Usage

1. Upload a photo of waste (drag & drop or click to browse)
2. Enter classification labels (one per line)
3. Click **Classify** to get results with confidence scores and recycling tips

## Default Categories

- cardboard, glass, metal, paper, plastic, trash

## Optimization

To quantize the model for CPU performance:

```bash
python optimize.py
python optimize.py --verify
```

## Tech Stack

- **Model**: OpenAI CLIP (`openai/clip-vit-base-patch32`)
- **Interface**: Gradio
- **ML**: PyTorch, Transformers
