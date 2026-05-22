"""
Zero-Shot Interactive Trash Sorter - Gradio Web UI
Uses CLIP for zero-shot waste classification with custom text labels.
"""

import gradio as gr
from PIL import Image
import torch

from engine import TrashClassifier, get_recycling_tip

DEFAULT_LABELS = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

classifier = TrashClassifier(quantized=False)


def classify_image(image: Image.Image, labels_text: str, top_k: int = 3):
    """
    Classify an image against user-provided text labels.

    Args:
        image: PIL Image uploaded by user.
        labels_text: Newline-separated text labels.
        top_k: Number of top results to display.

    Returns:
        Tuple of (waste identification markdown, bar chart data, recycling tip).
    """
    if image is None:
        return "## Waste Identified\nPlease upload an image.", None, "Please upload an image."

    label_list = [l.strip() for l in labels_text.strip().split("\n") if l.strip()]

    if not label_list:
        return "## Waste Identified\nPlease enter at least one label.", None, "Please enter labels."

    try:
        results = classifier.predict(image, label_list, top_k=top_k)
    except Exception as e:
        return "## Classification Error", None, f"**Error:** {str(e)}"

    top_label = results[0]["label"] if results else "unknown"
    top_confidence = results[0]["confidence"] if results else 0

    # Create prominent waste identification message
    waste_id_text = f"# {top_label.upper()}\n## Confidence: {top_confidence:.1f}%"

    tip = get_recycling_tip(top_label)

    chart_data = {
        "labels": [r["label"] for r in results],
        "values": [r["confidence"] for r in results],
    }

    return waste_id_text, chart_data, tip


css = """
.gradio-container {max-width: 1200px !important; margin: auto !important;}
.prediction-card {background: #f8f9fa; border-radius: 12px; padding: 20px; margin-top: 15px;}
.tip-box {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; padding: 20px; margin-top: 15px;}
.tip-box h3 {color: white; margin-bottom: 10px;}
.chart-container {margin-top: 20px;}
.waste-identified {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; border-radius: 12px; padding: 20px; text-align: center;}
.waste-identified h1 {color: white; margin-bottom: 5px;}
.waste-identified p {color: white; font-size: 1.2em;}
"""

with gr.Blocks(title="Zero-Shot Trash Sorter") as demo:
    gr.Markdown("# Zero-Shot Interactive Trash Sorter")
    gr.Markdown("Upload an image and enter custom text labels for zero-shot classification using CLIP.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Waste Image",
                type="pil",
                height=400,
            )
            labels_input = gr.Textbox(
                label="Classification Labels (one per line)",
                placeholder="cardboard\nglass\nmetal\npaper\nplastic\ntrash",
                lines=6,
                value="\n".join(DEFAULT_LABELS),
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Top K Results",
            )
            classify_btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=1):
            waste_id_output = gr.Markdown(
                "## Waste Identified\nUpload an image and click **Classify** to see results.",
                elem_classes=["waste-identified"],
            )
            chart_output = gr.BarPlot(
                label="Classification Results",
                x="labels",
                y="values",
                height=350,
                color="steelblue",
            )
            tip_output = gr.Markdown(
                "Upload an image and click **Classify** to get a recycling tip.",
                elem_classes=["tip-box"],
            )

    gr.Examples(
        examples=[["examples/soda_can.jpg", "\n".join(DEFAULT_LABELS), 3]],
        inputs=[image_input, labels_input, top_k_slider],
    )

    classify_btn.click(
        fn=classify_image,
        inputs=[image_input, labels_input, top_k_slider],
        outputs=[waste_id_output, chart_output, tip_output],
    )

if __name__ == "__main__":
    demo.launch(css=css)
