import os
from typing import List, Tuple
from uuid import uuid4

import gradio as gr
import pandas as pd
from langchain_core.runnables.base import RunnableSequence

from ..config import config
from ..evaluators.chains import get_eval_chain
from ..evaluators.post_analysis import calculate_fit_scores
from ..models.input_models import CandidateEvaluationWeights, InputModel
from ..preprocessing.parsers.pdf_parser import process_pdfs
from ..utils.helper import read_job_data, save_upload_file
from ..utils.logger import get_logger
from ..utils.process_jobs import process_all_jobs, process_all_pairs

logger = get_logger(__name__)


def process_input(
    text_input,
    additional_text,
    file_upload,
    input_type,
    api_key,
    interface,
    model,
    technical_skills,
    soft_skills,
    experience,
    education,
):

    try:

        logger.info("Starting processing input data.")

        # Validate input
        weights = CandidateEvaluationWeights(
            technical_skills=technical_skills,
            soft_skills=soft_skills,
            experience=experience,
            education=education,
        )
        input_data = InputModel(
            text_input=text_input,
            additional_text=additional_text,
            input_type=input_type,
            api_key=api_key,
            interface=interface,
            model=model,
            weights=weights,
        )
    except ValueError as e:
        logger.error(f"process_input: Error validating input: {str(e)}")
        return pd.DataFrame()

    # JD EVALUATION

    # get jd eval chain
    logger.info("Starting JD evaluation.")
    jd_grader_tuple = get_eval_chain(
        input_data.interface,
        input_data.model,
        os.getenv("GROQ_API_KEY"),
        eval_type="jd",
    )
    process_job_description(input_data, jd_grader_tuple)

    logger.info("Starting CV evaluation.")

    cv_data = process_cv_data(input_data, file_upload)
    job_data = read_job_data()
    job_tuples = pd.read_csv(f"{config.CSV_OUTPUT_DIR}/job_tuples.csv")
    cv_grader_tuple = get_eval_chain(
        input_data.interface,
        input_data.model,
        os.getenv("GROQ_API_KEY"),
        eval_type="cv",
    )
    evaluate_cv(cv_grader_tuple, job_data, cv_data)

    eval_results = calculate_and_save_fit_scores(
        input_data, cv_data, job_tuples, job_data
    )

    logger.info(
        f"processing completed. results saved in : {config.CSV_OUTPUT_DIR}, results type: {type(eval_results)}"
    )
    return eval_results


def process_job_description(
    input_data: InputModel, jd_grader_tuple: Tuple[str, RunnableSequence]
) -> None:
    """process the job description"""

    logger.info("Processing all jobs.")

    process_all_jobs(
        model_tuples=jd_grader_tuple,
        job_text=input_data.text_input,
        output_dir=config.JOBS_OUTPUT_DIR,
    )


def process_cv_data(input_data: InputModel, file_upload: List[gr.FileData]) -> None:
    """process the cv data"""

    logger.info("Processing all CVs.")

    if input_data.input_type == "Text" and input_data.additional_text:
        return [(str(uuid4()), input_data.additional_text)]
    elif input_data.input_type == "File" and file_upload is not None:
        try:
            for file in file_upload:
                if file.name.endswith(".pdf"):
                    save_upload_file(file)
            cv_data = process_pdfs(config.PDF_UPLOAD_FOLDER)
            cv_data = [(str(uuid4()), cv) for cv in cv_data]
            return cv_data
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise e


def calculate_and_save_fit_scores(
    input_data: InputModel,
    cv_data: List[Tuple[str, str]],
    job_tuple: List[Tuple[str, dict]],
    job_data: List[Tuple[str, dict]],
) -> pd.DataFrame:
    """Calculate the fit scores"""
    fit_scores_df = calculate_fit_scores(config.CV_OUTPUT_DIR, input_data.weights)

    cv_df = pd.DataFrame(cv_data, columns=["cv_id", "cv_text"])
    jd_df = pd.DataFrame(job_tuple, columns=["job_id", "job_text"])
    job_df = pd.DataFrame(job_data, columns=["job_id", "job_analysis"])

    fit_scores_df = pd.merge(fit_scores_df, jd_df, on="job_id", how="left")
    fit_scores_df = pd.merge(fit_scores_df, cv_df, on="cv_id", how="left")
    fit_scores_df = pd.merge(fit_scores_df, job_df, on="job_id", how="left")
    fit_scores_df.to_csv(
        f"{config.CSV_OUTPUT_DIR}/fit_scores_with_text.csv", index=False
    )
    logger.info("Fit scores calculated and saved.")
    return fit_scores_df


def evaluate_cv(
    cv_grader_tuple: Tuple[str, RunnableSequence],
    job_data: List[Tuple[str, dict]],
    cv_data: List[Tuple[str, str]],
) -> pd.DataFrame:
    """Evaluate the CVs"""
    logger.info("Evaluating all CVs.")
    process_all_pairs(
        cv_grader_tuple, job_data, cv_data, output_dir=config.CV_OUTPUT_DIR
    )
