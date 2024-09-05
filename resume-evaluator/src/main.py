from __future__ import annotations

from .app import create_gradio_app


if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch()
