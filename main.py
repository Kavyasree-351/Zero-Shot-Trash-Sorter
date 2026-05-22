"""
Zero-Shot Trash Sorter - Main Entry Point
Launches the Gradio web interface for CLIP-based waste classification.
"""

import sys


def main():
    """Run the Gradio application."""
    from app import demo, css

    print("Starting Zero-Shot Trash Sorter...")
    print("Open http://localhost:7860 in your browser")
    demo.launch(server_name="0.0.0.0", server_port=7860, css=css)


if __name__ == "__main__":
    main()
