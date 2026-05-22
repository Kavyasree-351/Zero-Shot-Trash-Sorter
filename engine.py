"""
Zero-Shot Trash Classifier Engine
Uses OpenAI's CLIP model for zero-shot image classification with dynamic text labels.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class TrashClassifier:
    """Zero-shot image classifier using CLIP for waste categorization."""

    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self, quantized: bool = False):
        """
        Initialize the CLIP model and processor.

        Args:
            quantized: If True, loads a quantized INT8 model for reduced memory usage.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(
            self.MODEL_NAME,
            use_safetensors=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)

        if quantized:
            self._apply_quantization()

    def _apply_quantization(self):
        """Apply INT8 quantization to linear layers for reduced memory footprint."""
        from torch.quantization import quantize_dynamic

        self.model = quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

    def predict(
        self,
        image: Image.Image,
        candidate_labels: list[str],
        top_k: int = 3
    ) -> list[dict]:
        """
        Classify an image using zero-shot learning with dynamic text labels.

        Args:
            image: PIL Image to classify.
            candidate_labels: List of text descriptions for classification.
            top_k: Number of top results to return.

        Returns:
            List of dicts with 'label', 'confidence', and 'score' keys, sorted by confidence.
        """
        if not candidate_labels:
            raise ValueError("candidate_labels cannot be empty")

        if len(candidate_labels) == 1:
            candidate_labels = [candidate_labels[0], "other"]

        inputs = self.processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        results = []
        for i, label in enumerate(candidate_labels):
            results.append({
                "label": label,
                "confidence": float(probs[0][i].item() * 100),
                "score": float(probs[0][i].item())
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]

    def get_embedding(self, image: Image.Image) -> torch.Tensor:
        """Get the image embedding vector for advanced use cases."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features


RECYCLING_TIPS = {
    "cardboard": "Flatten boxes and remove any plastic packaging. Keep dry and clean.",
    "glass": "Rinse thoroughly and remove caps. Broken glass should go in trash for safety.",
    "metal": "Empty and rinse cans. Aluminum foil should be clean and balled up.",
    "paper": "Keep dry and separate from food waste. Shredded paper may need special handling.",
    "plastic": "Check the resin number. Rinse and remove caps when possible.",
    "trash": "When in doubt, throw it out. Contaminated recyclables hurt the whole batch.",
    "organic": "Compost food scraps and yard waste. Keep out of recycling streams.",
    "hazardous": "Batteries, electronics, and chemicals need special disposal - never in regular trash.",
}


def get_recycling_tip(label: str) -> str:
    """Get a recycling tip based on the predicted label."""
    label_lower = label.lower()

    for key, tip in RECYCLING_TIPS.items():
        if key in label_lower:
            return tip

    return "When in doubt, check your local recycling guidelines. Rules vary by location."
