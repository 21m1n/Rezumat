
# Resume Evaluator

## Overview

Rézumat leverages LLMs to evaluate candidate resumes against job descriptions. The project uses Gradio for the user interface.

## Project Structure

```
resume-evaluator/
├── data/
│   ├── input/
│   │   └── pdf/
│   └── output/
│       ├── csv/
│       ├── cv/
│       └── jobs/
├── logs/
├── src/
│   ├── evaluators/
│   ├── models/
│   ├── preprocessing/
│   │   └── parsers/
│   ├── prompts/
│   ├── utils/
│   ├── app.py
│   ├── config.py
│   └── main.py
├── tests/
├── Dockerfile
└── requirements.txt
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/resume-evaluator.git
   cd resume-evaluator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Gradio application:
   ```
   python -m resume-evaluation.src.main 
   ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:7860`).

3. Upload a resume PDF and enter a job description in the interface.

4. Click the "Evaluate" button to process the resume and get the evaluation results.

## Docker Support

To run the application using Docker:

1. Build the Docker image:
   ```
   docker build -t resume-evaluator .
   ```

2. Run the container:
   ```
   docker run -p 7860:7860 resume-evaluator
   ```

## Testing

Run the tests using:
```
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.