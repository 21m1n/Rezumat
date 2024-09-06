from __future__ import annotations

from rezumat.config import config  # to ensure all setup is done
from rezumat.app import create_gradio_app


def main():
    demo = create_gradio_app()
    demo.launch()


if __name__ == "__main__":
    main()
