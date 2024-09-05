# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /

# Copy the current directory contents into the container at /resume-evaluator
COPY . /resume-evaluator

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /resume-evaluator/requirements.txt

# Create necessary directories
RUN mkdir -p /resume-evaluator/resume-evaluator/data/input/pdf \
    /resume-evaluator/resume-evaluator/data/output/csv \
    /resume-evaluator/resume-evaluator/data/output/jobs \
    /resume-evaluator/resume-evaluator/data/output/cv \
    /resume-evaluator/resume-evaluator/logs

# Create an empty log file
RUN touch /resume-evaluator/resume-evaluator/logs/evaluation_log.txt

# Set environment variable for Python to use UTF-8 encoding
ENV PYTHONIOENCODING=utf-8

# Set the PYTHONPATH to include the resume-evaluator directory
ENV PYTHONPATH=/resume-evaluator:$PYTHONPATH

ENV GRADIO_SERVER_NAME="0.0.0.0"

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Ensure correct permissions
RUN chown -R root:root /resume-evaluator
RUN chmod -R 755 /resume-evaluator

# Run main.py when the container launches
CMD ["python", "-m", "resume-evaluator.src.main"]
