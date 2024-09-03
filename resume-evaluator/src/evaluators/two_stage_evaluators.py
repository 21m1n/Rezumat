import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from langchain_core.runnables import RunnableSequence
from tqdm import tqdm


def two_stage_eval_jd(model_tuples: List[Tuple[str, RunnableSequence]], job_tuple: Tuple[str, str], output_dir: str) -> Union[pd.DataFrame, None]:
    model_results = {}
    
    if isinstance(model_tuples, Tuple):
        model_tuples = [model_tuples]
    
    job_id, job_description = job_tuple
    
    for model_name, grader in model_tuples:
        try:
            result = grader.invoke({"job_description": job_description})
            model_results[model_name] = result

            # save model result 
            json_file = os.path.join(output_dir, f"{job_id}_{model_name}.json")
            with open(json_file, "w") as f:
                json.dump(result, f, indent=4)
                logging.info(f"Saved {model_name} result for job_id: {job_id}")
            time.sleep(2.1)  # Add a small delay to avoid rate limiting

        except Exception as e:
            error_msg = f"Error with {model_name} for job_id: {job_id}. Error: {str(e)}"
            logging.error(error_msg)
            print(error_msg)

    if not model_results:
        error_msg = f"All models failed for job_id: {job_id}."
        logging.error(error_msg)
        print(error_msg)
        return None

def two_stage_eval_cv(model_tuples: List[Tuple[str, RunnableSequence]], job_tuple: Tuple[str, str], cv_tuple: Tuple[str, str], output_dir: str) -> Union[pd.DataFrame, None]:
    model_results = {}
    
    if isinstance(model_tuples, Tuple):
        model_tuples = [model_tuples]   
        
    job_id, job_requirements = job_tuple
    cv_id, cv = cv_tuple
    
    for model_name, grader in model_tuples:
        try:
            result = grader.invoke({"job_requirements": job_requirements, "resume": cv})
            model_results[model_name] = result

            # save model result 
            json_file = os.path.join(output_dir, f"{job_id}_{cv_id}_{model_name}.json")
            with open(json_file, "w") as f:
                json.dump(result, f, indent=4)
            time.sleep(2.1)  # Add a small delay to avoid rate limiting

        except Exception as e:
            error_msg = f"Error with {model_name} for job_id: {job_id}. Error: {str(e)}"
            logging.error(error_msg)
            print(error_msg)

    if not model_results:
        error_msg = f"All models failed for job_id: {job_id}."
        logging.error(error_msg)
        print(error_msg)
        return None

def resume_evaluation(eval_results_folder: str) -> pd.DataFrame:

    results = []
    errors = []

    for file in tqdm(Path(eval_results_folder).glob("*.json"), total=len(list(Path(eval_results_folder).glob("*.json"))), desc="Evaluating results"):
        try:
            file_name = file.stem
            job_id, cv_id, model_name = file_name.split("_")
            with open(file, "r") as f:
                result = json.load(f)
            
            data = {
                "job_id": job_id,
                "cv_id": cv_id,
                "model_name": model_name,
                "original_technical_skills": result["resume_evaluation"]["original_scores"].get("technical_skills", None),
                "original_soft_skills": result["resume_evaluation"]["original_scores"].get("soft_skills", None),
                "original_experience": result["resume_evaluation"]["original_scores"].get("experience", None),
                "original_education": result["resume_evaluation"]["original_scores"].get("education", None),
                "recalibrated_technical_skills": result["recalibrated_scores"].get("technical_skills", None),
                "recalibrated_soft_skills": result["recalibrated_scores"].get("soft_skills", None),
                "recalibrated_experience": result["recalibrated_scores"].get("experience", None),
                "recalibrated_education": result["recalibrated_scores"].get("education", None),
                "inferred_experience": ", ".join(result["deeper_analysis"].get("inferred_experience", [])),
                "suitability": result["assessment"].get("suitability", None),
                "strengths": result["assessment"].get("strengths", None),
                "concerns": result["assessment"].get("concerns", None)
            }
            
            results.append(data)
        except Exception as e:
            errors.append({"file": str(file), "error": str(e)})
            print(f"Error processing {file}: {e}")

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    
    return df

def calculate_fit_scores(eval_results_folder: str, weights: Dict[str, float]) -> pd.DataFrame:

    df = resume_evaluation(eval_results_folder)
    
    weights = {
    "technical_skills": weights.technical_skills,
    "soft_skills": weights.soft_skills,
    "experience": weights.experience,
    "education": weights.education
    }
    
    # Define score types
    score_types = ["original", "recalibrated"]
    
    # Calculate overall scores using pandas' dot product
    for score_type in score_types:
        columns = [f"{score_type}_{skill}" for skill in weights.keys()]
        df[score_type+"_overall_score"] = df[columns].values.dot(pd.Series(weights).values)
    
    return df