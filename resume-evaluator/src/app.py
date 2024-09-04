from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union
from uuid import uuid4

import gradio as gr
import pandas as pd
from langchain_core.runnables.base import RunnableSequence

from .config import config
from .evaluators.chains import get_eval_chain
from .evaluators.post_analysis import calculate_fit_scores
from .models.input_models import CandidateEvaluationWeights, InputModel
from .preprocessing.parsers.pdf_parser import process_pdfs
from .utils.process_jobs import process_all_jobs, process_all_pairs

# Use config throughout the file
PDF_UPLOAD_FOLDER = config.PDF_UPLOAD_FOLDER
OUTPUT_DIR = config.OUTPUT_DIR
JOBS_OUTPUT_DIR = config.JOBS_OUTPUT_DIR
CSV_OUTPUT_DIR = config.CSV_OUTPUT_DIR
CV_OUTPUT_DIR = config.CV_OUTPUT_DIR
LOG_FILE = config.LOG_FILE

# set up log file 
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO, filename=config.LOG_FILE, datefmt="%Y-%m-%d %H:%M:%S")

def save_upload_file(file) -> None:
    """save the file uploaded by the user to the pdf upload folder PDF_UPLOAD_FOLDER"""
    if not os.path.exists(PDF_UPLOAD_FOLDER):
      os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)
    shutil.copy(file, PDF_UPLOAD_FOLDER)
    gr.Info(f"file is saved to {PDF_UPLOAD_FOLDER}{file.name.split('/')[-1]}")

def process_job_description(input_data: InputModel, jd_grader_tuple: Tuple[str, RunnableSequence]) -> None:
    """process the job description"""
    process_all_jobs(model_tuples=jd_grader_tuple, job_text=input_data.text_input, output_dir=JOBS_OUTPUT_DIR)

def process_cv_data(input_data: InputModel, file_upload: List[gr.FileData]) -> None:
    """process the cv data"""
    if input_data.input_type == "Text" and input_data.additional_text:
        return [(str(uuid4()),input_data.additional_text)]
    elif input_data.input_type == "File" and file_upload is not None:
        try:
            for file in file_upload:
                if file.name.endswith('.pdf'):
                    save_upload_file(file)
            cv_data = process_pdfs(PDF_UPLOAD_FOLDER)
            cv_data = [(str(uuid4()), cv) for cv in cv_data]
            return cv_data
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            raise e

# [TODO] to remove?
def read_job_data() -> List[Tuple[str, dict]]:
    """Read job data from JOBS_OUTPUT_DIR"""
    job_data = []
    for file in Path(JOBS_OUTPUT_DIR).glob("*.json"):
        job_id = file.stem.split("_")[0]
        with open(file, "r") as f:
            job_description = json.load(f)
            job_data.append((job_id, job_description))
    return job_data

def evaluate_cv(cv_grader_tuple: Tuple[str, RunnableSequence], job_data: List[Tuple[str, dict]], cv_data: List[Tuple[str, str]]) -> pd.DataFrame:
    """Evaluate the CVs"""
    process_all_pairs(cv_grader_tuple, job_data, cv_data, output_dir=CV_OUTPUT_DIR)
  
def calculate_and_save_fit_scores(input_data: InputModel, cv_data: List[Tuple[str, str]], job_data: List[Tuple[str, dict]]) -> pd.DataFrame:
    """Calculate the fit scores"""
    fit_scores_df = calculate_fit_scores(CV_OUTPUT_DIR, input_data.weights)
    
    cv_df = pd.DataFrame(cv_data, columns=["cv_id", "cv_text"])
    jd_df = pd.DataFrame(job_data, columns=["job_id", "job_text"])
    
    fit_scores_df = pd.merge(fit_scores_df, jd_df, on="job_id", how="left")
    fit_scores_df = pd.merge(fit_scores_df, cv_df, on="cv_id", how="left")
    fit_scores_df.to_csv(f"{CSV_OUTPUT_DIR}/fit_scores_with_text.csv", index=False)
    return fit_scores_df

def process_input(text_input, additional_text, file_upload, input_type, api_key, interface, model, technical_skills, soft_skills, experience, education):

    try:
        # Validate input
        weights = CandidateEvaluationWeights(
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
            weights=weights
        )
    except ValueError as e:
        logging.error(f"process_input: Error validating input: {str(e)}")
        return pd.DataFrame()


    # JD EVALUATION 
    
    # get jd eval chain 
    jd_grader_tuple = get_eval_chain(input_data.interface, input_data.model, os.getenv("GROQ_API_KEY"), eval_type="jd")
    process_job_description(input_data, jd_grader_tuple)
    
    cv_data = process_cv_data(input_data, file_upload)
    job_data = read_job_data()
    
    cv_grader_tuple = get_eval_chain(input_data.interface, input_data.model, os.getenv("GROQ_API_KEY"), eval_type="cv")
    evaluate_cv(cv_grader_tuple, job_data, cv_data)
    
    eval_results = calculate_and_save_fit_scores(input_data, cv_data, job_data)
    
    logging.info(f"processing completed. results saved in : {CSV_OUTPUT_DIR}, results type: {type(eval_results)}")
    return eval_results


# ------------------------------
# gradio interface 
# ------------------------------

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# ✏️ Resume Evaluator")
        
        # add a state to store the eval_results
        eval_results = gr.State()
        
        with gr.Group() as initial_view:
            with gr.Row():
                with gr.Column(scale=2):
                    # model selection section
                    api_key = gr.Textbox(label="API Key", type="password")
                    interface = gr.Dropdown(["Groq", "OpenAI", "Anthropic"], label="Interface", value="Groq")
                    model = gr.Dropdown(["llama3-70b-8192", "gpt-3.5-turbo", "gpt-4"], label="Model", value="llama3-70b-8192")
                    
                    # weights section
                    gr.Markdown("### Weightage (Total must be 100)")
                    technical_skills = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Technical Skills")
                    soft_skills = gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Soft Skills")
                    experience = gr.Slider(minimum=0, maximum=100, value=20, step=1, label="Experience")
                    education = gr.Slider(minimum=0, maximum=100, value=10, step=1, label="Education")
                
                with gr.Column(scale=8):
                  
                    # Main content (80% of the screen)
                    
                    # # initial view 
                    # with gr.Group() as initial_view:
                        
                    # job description section
                    jd_text_input = gr.TextArea(label="Job Description", lines=12, info="Paste the job description here")
                    input_type = gr.Radio(["Text", "File"], label="Select Resume Type", value="Text", info="Select the type of input") 
                    
                    # resume section
                    with gr.Row() as additional_input_row:
                        additional_text = gr.Textbox(label="Resume", lines=5, visible=True, info="Paste the resume here")
                        file_upload = gr.File(label="Upload File", file_count="multiple", file_types=["pdf"], visible=False, )
                    
                    # output = gr.Markdown(label="Output")
        
                    with gr.Row():
                        submit_btn = gr.Button("Evaluate")
                        reset_btn = gr.Button("Reset")
              
        # results view (initially hidden)
        with gr.Group(visible=False) as results_view:
          
            with gr.Row():
                total_applicants = gr.Number(label="Total Applicants")
                yes_count = gr.Number(label="Yes")
                no_count = gr.Number(label="No")
                kiv_count = gr.Number(label="KIV")
                
            with gr.Row():
                top_candidates = gr.Dropdown(label="Top Candidates")
            
            with gr.Row():
                jd_display = gr.TextArea(label="Job Description", interactive=False)
                cv_display = gr.TextArea(label="Selected CV", interactive=False)
              
            with gr.Row():
                strengths = gr.TextArea(label="Strengths", interactive=False)
                concerns = gr.TextArea(label="Concerns", interactive=False)
            
            back_btn = gr.Button("Back")
                    
                
        def update_input_type(choice):
          if choice == "Text":
              return gr.update(visible=True), gr.update(visible=False)
          else:
              return gr.update(visible=False), gr.update(visible=True)
            
        def reset_interface():
            return (
                "", "Text", "", None, "", 50, 20, 20, 10
            ) 
    
        def process_results(results_df: Union[pd.DataFrame, List[dict]]):
            
            logging.info("Processing results...")
            logging.info(f"DataFrame type: {results_df}")

            if results_df is None or results_df.empty:
                logging.warning("DataFrame is None or empty")
                return [
                  gr.update(visible=True),   # Keep initial view visible
                  gr.update(visible=False),  # Keep results view hidden
                  None, None, None, None,    # Total, Yes, No, KIV counts
                  None,                      # Top candidates dropdown
                  "",                        # Job description
                  None,                      # Eval results state
                  "Error: No results received from process_input function"  # Debug output
                ]
                  
            if not isinstance(results_df, pd.DataFrame):
              logging.error(f"Expected a pandas DataFrame, but got {type(results_df)}")
              return [
                gr.update(visible=True),   # Keep initial view visible
                gr.update(visible=False),  # Keep results view hidden
                None, None, None, None,    # Total, Yes, No, KIV counts
                None,                      # Top candidates dropdown
                "",                        # Job description
                None,                      # Eval results state
                "Error: DataFrame expected"  # Debug output
              ]
                  
            logging.info(f"DataFrame shape: {results_df.shape}")
            logging.info(f"DataFrame columns: {results_df.columns}")
              
            try:              
                total = len(results_df)
                yes = len(results_df[results_df["suitability"] == "yes"])
                no = len(results_df[results_df["suitability"] == "no"])
                kiv = len(results_df[results_df["suitability"] == "kiv"])
                
                top_candidates = results_df[results_df["suitability"].isin(["yes", "kiv"])].sort_values(by="recalibrated_overall_score", ascending=False).head(5)
                top_candidates_list = top_candidates["cv_text"].tolist()
                
                job_description = results_df["job_text"].iloc[0] if not results_df.empty else ""
                
                debug_msg = f"Total: {total}, Yes: {yes}, No: {no}, KIV: {kiv}, Top candidates: {top_candidates_list}"
                logging.info(debug_msg)
                
                return (
                    gr.update(visible=False), # hide initial view 
                    gr.update(visible=True),  # show results  
                    total, yes, no, kiv, 
                    gr.Dropdown(choices=top_candidates_list, value=top_candidates_list[0] if top_candidates_list else None),
                    job_description,
                    results_df # store full results in state 
                )
            except Exception as e:
                error_msg = f"Error processing results: {str(e)}"
                logging.error(error_msg)
                return [
                    gr.update(visible=True),   # Keep initial view visible
                    gr.update(visible=False),  # Keep results view hidden
                    None, None, None, None,    # Total, Yes, No, KIV counts
                    None,                      # Top candidates dropdown
                    "",                        # Job description
                    None,                      # Eval results state
                    error_msg                  # Debug output
                ]
            
        def display_candidate_info(cv_id, results_df):

            if results_df is None or results_df.empty:
                return gr.update(value="No candidates found"), gr.update(value=""), gr.update(value="")
            matching_candidates = results_df[results_df["cv_id"] == cv_id]
            
            if matching_candidates.empty:
                logging.warning(f"No matching candidates found for cv_id: {cv_id}")
                return "No matching candidates found", "", ""
            
            candidate_info = matching_candidates.iloc[0]
            
            return (
                candidate_info["cv_text"],
                candidate_info.get("strengths", "No strengths information available"),
                candidate_info.get("concerns", "No concerns information available")
            )
        
        input_type.change(update_input_type, inputs=[input_type], outputs=[additional_text, file_upload])
        
        submit_btn.click(
            fn=process_input,
            inputs=[jd_text_input, additional_text, file_upload, input_type, api_key, interface, model, technical_skills, soft_skills, experience, education],
            outputs=eval_results
        ).then(
          fn=process_results,
          inputs=[eval_results],
          outputs=[initial_view, results_view, total_applicants, yes_count, no_count, kiv_count, top_candidates, jd_display, eval_results]
        )
    
        reset_btn.click(
            fn=reset_interface,
            inputs=[],
            outputs=[jd_text_input, input_type, additional_text, file_upload, initial_view, results_view, technical_skills, soft_skills, experience, education]
        )
        
        top_candidates.change(
          fn=display_candidate_info,
          inputs=[top_candidates, eval_results],
          outputs=[cv_display, strengths, concerns] 
        )

        back_btn.click(
          fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
          inputs=[],
          outputs=[initial_view, results_view]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()