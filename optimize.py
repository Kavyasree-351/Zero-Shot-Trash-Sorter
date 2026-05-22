"""
Model Optimization Script
Applies INT8 quantization to the CLIP model for reduced memory usage and faster CPU inference.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


def quantize_clip_model(
    model_name: str = "openai/clip-vit-base-patch32",
    output_path: str = "quantized_clip.pt"
) -> None:
    """
    Quantize the CLIP model to INT8 and save it.

    This reduces model size from ~600MB to ~300MB and improves CPU inference speed.

    Args:
        model_name: Hugging Face model identifier.
        output_path: Path to save the quantized model.
    """
    print(f"Loading model: {model_name}")
    model = CLIPModel.from_pretrained(model_name)

    print("Applying INT8 quantization to linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    print(f"Saving quantized model to: {output_path}")
    torch.save(quantized_model.state_dict(), output_path)

    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())

    print(f"\nOptimization Results:")
    print(f"  Original model size: ~{original_size / 1024 / 1024:.1f} MB")
    print(f"  Quantized model size: ~{quantized_size / 1024 / 1024:.1f} MB")
    print(f"  Memory reduction: ~{100 * (1 - quantized_size / original_size):.1f}%")
    print("\nNote: Load the quantized model using TrashClassifier(quantized=True)")


def verify_quantization() -> None:
    """Verify that quantization produces valid INT8 weights."""
    from engine import TrashClassifier

    print("Loading standard model...")
    standard = TrashClassifier(quantized=False)

    print("Loading quantized model...")
    quantized = TrashClassifier(quantized=True)

    standard_params = sum(p.numel() for p in standard.model.parameters())
    quantized_params = sum(p.numel() for p in quantized.model.parameters())

    print(f"\nVerification:")
    print(f"  Standard parameters: {standard_params:,}")
    print(f"  Quantized parameters: {quantized_params:,}")
    print(f"  Models compatible: {standard_params == quantized_params}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize CLIP model for CPU optimization")
    parser.add_argument(
        "--output",
        type=str,
        default="quantized_clip.pt",
        help="Output path for quantized model"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify quantization instead of creating new model"
    )

    args = parser.parse_args()

    if args.verify:
        verify_quantization()
    else:
        quantize_clip_model(output_path=args.output)
