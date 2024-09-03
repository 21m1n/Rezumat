import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from uuid import uuid4

import gradio as gr
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pypdf import PdfReader
from tqdm import tqdm
from tqdm.auto import tqdm

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from .evaluation.resume_evaluator import (two_stage_eval_cv,
                                                           two_stage_eval_jd)
from .prompts.two_stage_eval_cv import \
    TWO_STAGE_EVAL_CV_PROMPT
from .prompts.two_stage_eval_jd import \
    TWO_STAGE_EVAL_JD_PROMPT

# Global variables
OUTPUT_PATH = Path("./output")
LOG_FILE_PATH = Path("./logs")
ENV_PATH = "../../.env"

NUM_WORKERS = os.cpu_count()
TEMPERATURE = 0.0
MAX_TOKENS = 8192
GROQ_MODEL = "llama3-70b-8192"

# load environment variables
load_dotenv(find_dotenv(ENV_PATH))

# get current time
current_time = datetime.now().strftime(("%Y%m%d_%H%M"))

# log file
log_file = os.path.join(LOG_FILE_PATH, "evaluating_log.txt")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, filename=log_file)

# Set up the Groq model and prompts
groq_llm = ChatGroq(model=GROQ_MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

jd_eval_prompt = PromptTemplate(
    input_variables=["job_description"],
    template=TWO_STAGE_EVAL_JD_PROMPT
)

cv_eval_prompt = PromptTemplate(
    input_variables=["job_requirements", "resume"],
    template=TWO_STAGE_EVAL_CV_PROMPT
)

groq_jd_grader = jd_eval_prompt | groq_llm | JsonOutputParser()
groq_cv_grader = cv_eval_prompt | groq_llm | JsonOutputParser()

model_tuples = [("groq_jd_grader", groq_jd_grader)]
cv_model_tuples = [("groq_cv_grader", groq_cv_grader)]

def parse_pdf(file) -> str:
    reader = PdfReader(file)
    return " ".join([page.extract_text() for page in reader.pages])

def process_results(results: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    return df

def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    # Implement your scoring logic here
    # For example:
    df['recalibrated_overall_score'] = df['overall_score'] * 0.7 + df['technical_skills_score'] * 0.3
    return df

def evaluate_resumes(job_description: str, resumes: List[gr.File]) -> str:
    OUTPUT_PATH.mkdir(exist_ok=True)
    job_id = str(uuid4())

    logging.info(f"Starting evaluation for job ID: {job_id}")

    # Evaluate job description
    job_requirements = two_stage_eval_jd(model_tuples, job_description, job_id, OUTPUT_PATH)
    logging.info(f"Job requirements extracted for job ID: {job_id}")

    # Evaluate resumes
    results = []
    for resume in tqdm(resumes, desc="Evaluating resumes"):
        cv_id = str(uuid4())
        try:
            cv_text = parse_pdf(resume.name)
            result = two_stage_eval_cv(cv_model_tuples, json.dumps(job_requirements), job_id, cv_text, cv_id, OUTPUT_PATH)
            results.append(result)
        except Exception as e:
            logging.error(f"Error processing resume {resume.name}: {str(e)}")

    logging.info(f"Completed evaluation of {len(results)} resumes for job ID: {job_id}")

    # Process results
    df = process_results(results)
    df = calculate_scores(df)


    # Save results to CSV
    output_file = OUTPUT_PATH / f"evaluation_results_{job_id}.csv"
    df.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")

    return str(output_file)

# Create the Gradio interface
iface = gr.Interface(
    fn=evaluate_resumes,
    inputs=[
        gr.Textbox(label="Job Description", lines=10),
        gr.File(label="Resumes", file_count="multiple")
    ],
    outputs=gr.Textbox(label="Evaluation Results"),
    title="Resume Evaluation App",
    description="Upload a job description and multiple resumes to evaluate and shortlist potential candidates."
)

if __name__ == "__main__":
    iface.launch()