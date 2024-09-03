import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

import gradio as gr
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm

from .config import config
from .evaluators.chains import get_eval_chain
from .evaluators.post_analysis import calculate_fit_scores
from .preprocessing.parsers.pdf_parser import process_pdfs
from .utils.process_jobs import process_all_jobs, process_all_pairs

# Use config throughout the file
PDF_UPLOAD_FOLDER = config.PDF_UPLOAD_FOLDER
OUTPUT_DIR = config.OUTPUT_DIR
JOBS_OUTPUT_DIR = config.JOBS_OUTPUT_DIR
CV_OUTPUT_DIR = config.CV_OUTPUT_DIR
LOG_FILE = config.LOG_FILE

# set up log file 
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, filename=config.LOG_FILE, datefmt="%Y-%m-%d %H:%M:%S")

def save_upload_file(file):
  """save the file to the pdf upload folder"""
  if not os.path.exists(PDF_UPLOAD_FOLDER):
    os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)
  shutil.copy(file, PDF_UPLOAD_FOLDER)
  gr.Info(f"file is saved to {PDF_UPLOAD_FOLDER}{file.name.split('/')[-1]}")


class WeightageModel(BaseModel):
    technical_skills: int = Field(ge=0, le=100)
    soft_skills: int = Field(ge=0, le=100)
    experience: int = Field(ge=0, le=100)
    education: int = Field(ge=0, le=100)

    @model_validator(mode='after')
    def check_total(self) -> 'WeightageModel':
        total = self.technical_skills + self.soft_skills + self.experience + self.education
        if total != 100:
            raise ValueError(f"Total weightage must be 100%. Current total: {total}%")
        return self

class InputModel(BaseModel):
    text_input: str
    additional_text: str = ""
    input_type: Literal["Text", "File"]
    api_key: str
    interface: Literal["Groq", "OpenAI", "Anthropic"]
    model: Literal["llama3-70b-8192", "gpt-3.5-turbo", "gpt-4"]
    weightage: WeightageModel

def process_input(text_input, additional_text, file_upload, input_type, api_key, interface, model, technical_skills, soft_skills, experience, education):
    content = ""
    try:
        # Validate input
        weightage = WeightageModel(
            technical_skills=technical_skills,
            soft_skills=soft_skills,
            experience=experience,
            education=education
        )
        input_data = InputModel(
            text_input=text_input,
            additional_text=additional_text,
            input_type=input_type,
            api_key=api_key,
            interface=interface,
            model=model,
            weightage=weightage
        )
    except ValueError as e:
        return f"Validation Error: {str(e)}"


    # JD EVALUATION 
    
    # get the job description 
    if input_data.text_input:
        job_description = input_data.text_input
    
    # get jd eval chain 
    jd_grader_tuple = get_eval_chain(input_data.interface, input_data.model, os.getenv("GROQ_API_KEY"), eval_type="jd")
    process_all_jobs(jd_grader_tuple, job_description, output_dir=JOBS_OUTPUT_DIR)

    # CV EVALUATION 

    # if the input type is text 
    if input_data.input_type == "Text" and input_data.additional_text:
        cv_data = [(str(uuid4()),input_data.additional_text)]
    # if the input type is an uploaded file 
    elif input_data.input_type == "File" and file_upload is not None:
        try:
            # upload and save the files
            for file in file_upload:
                if file.name.endswith('.pdf'):
                    save_upload_file(file)
            # parse the pdfs
            cv_data = process_pdfs(PDF_UPLOAD_FOLDER)
            cv_data = [(str(uuid4()), cv_data[i]) for i in range(len(cv_data))]
        except Exception as e:
            content += f"Error processing file: {str(e)}\n\n" 

    logging.info(f"reading job data from {JOBS_OUTPUT_DIR}")
    job_data = []
    for file in Path(JOBS_OUTPUT_DIR).glob("*.json"):
            file_name = file.stem
            job_id = file_name.split("_")[0]
            model_name = file_name.split("_")[1]
            with open(file, "r") as f:
                job_description = json.load(f)
                job_data.append((job_id, job_description))

    logging.info("evaluating CV now.")
    # get the cv eval chain 
    cv_grader_tuple = get_eval_chain(input_data.interface, input_data.model, os.getenv("GROQ_API_KEY"), eval_type="cv")
    process_all_pairs(cv_grader_tuple, job_data, cv_data, output_dir=CV_OUTPUT_DIR)
    logging.info("CV evaluation done.") 

    # calculate fit scores 
    fit_scores_df = calculate_fit_scores(CV_OUTPUT_DIR, input_data.weightage)
    print(fit_scores_df["suitability"].value_counts())
    
    return content




# ------------------------------
# gradio interface 
# ------------------------------
def reset_interface():
    return (
        "", "Text", "", None, "", 50, 20, 20, 10
    )
    
with gr.Blocks() as demo:
    gr.Markdown("# ✏️ Resume Evaluator")
    
    with gr.Row():
        with gr.Column(scale=2):
          
            # model selection section
            api_key = gr.Textbox(label="API Key", type="password")
            interface = gr.Dropdown(["Groq", "OpenAI", "Anthropic"], label="Interface", value="Groq")
            model = gr.Dropdown(["llama3-70b-8192", "gpt-3.5-turbo", "gpt-4"], label="Model", value="llama3-70b-8192")
            
            # weightage section
            gr.Markdown("### Weightage (Total must be 100)")
            technical_skills = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Technical Skills")
            soft_skills = gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Soft Skills")
            experience = gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Experience")
            education = gr.Slider(minimum=0, maximum=100, value=10, step=1, label="Education")
        
        with gr.Column(scale=8):
          
            # Main content (80% of the screen)
            
            # job description section
            jd_text_input = gr.TextArea(label="Job Description", lines=12, info="Paste the job description here")
            
            # input type section 
            input_type = gr.Radio(["Text", "File"], label="Select Resume Type", value="Text", info="Select the type of input")
            
            # resume section
            with gr.Row() as additional_input_row:
                additional_text = gr.Textbox(label="Resume", lines=5, visible=True, info="Paste the resume here")
                file_upload = gr.File(label="Upload File", file_count="multiple", file_types=["pdf"], visible=False, )
            
            output = gr.Markdown(label="Output")
    
            with gr.Row():
                submit_btn = gr.Button("Submit")
                reset_btn = gr.Button("Reset")
    
    def update_input_type(choice):
        if choice == "Text":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)
    
    input_type.change(update_input_type, inputs=[input_type], outputs=[additional_text, file_upload])
    
    submit_btn.click(
        fn=process_input,
        inputs=[jd_text_input, additional_text, file_upload, input_type, api_key, interface, model, technical_skills, soft_skills, experience, education],
        outputs=output
    )
    
    reset_btn.click(
        fn=reset_interface,
        inputs=[],
        outputs=[jd_text_input, input_type, additional_text, file_upload, output, technical_skills, soft_skills, experience, education]
    )

demo.launch()