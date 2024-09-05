import json
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

from ..utils.logger import get_logger

logger = get_logger(__name__)


def resume_evaluation(eval_results_folder: str) -> pd.DataFrame:

    logger.info("Start resume evaluation.")

    results = []
    errors = []

    for file in tqdm(
        Path(eval_results_folder).glob("*.json"),
        total=len(list(Path(eval_results_folder).glob("*.json"))),
        desc="Evaluating results",
    ):
        try:
            file_name = file.stem
            job_id, cv_id, model_name = file_name.split("_")
            with open(file, "r") as f:
                result = json.load(f)

            data = {
                "job_id": job_id,
                "cv_id": cv_id,
                "model_name": model_name,
                "original_technical_skills": result["resume_evaluation"][
                    "original_scores"
                ].get("technical_skills", None),
                "original_soft_skills": result["resume_evaluation"][
                    "original_scores"
                ].get("soft_skills", None),
                "original_experience": result["resume_evaluation"][
                    "original_scores"
                ].get("experience", None),
                "original_education": result["resume_evaluation"][
                    "original_scores"
                ].get("education", None),
                "recalibrated_technical_skills": result["recalibrated_scores"].get(
                    "technical_skills", None
                ),
                "recalibrated_soft_skills": result["recalibrated_scores"].get(
                    "soft_skills", None
                ),
                "recalibrated_experience": result["recalibrated_scores"].get(
                    "experience", None
                ),
                "recalibrated_education": result["recalibrated_scores"].get(
                    "education", None
                ),
                "inferred_experience": ", ".join(
                    result["deeper_analysis"].get("inferred_experience", [])
                ),
                "suitability": result["assessment"].get("suitability", None),
                "strengths": result["assessment"].get("strengths", None),
                "concerns": result["assessment"].get("concerns", None),
            }

            results.append(data)
        except Exception as e:
            errors.append({"file": str(file), "error": str(e)})
            logger.error(f"Error processing {file}: {e}")

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    return df


def calculate_fit_scores(
    eval_results_folder: str, weights: Dict[str, float]
) -> pd.DataFrame:

    logger.info(
        f"""Calculating fit scores based on the weights. "technical_skills": {weights.technical_skills}, 
        "soft_skills": {weights.soft_skills}, "experience": {weights.experience}, "education": {weights.education}."""
    )

    df = resume_evaluation(eval_results_folder)

    weights = {
        "technical_skills": weights.technical_skills,
        "soft_skills": weights.soft_skills,
        "experience": weights.experience,
        "education": weights.education,
    }

    # Define score types
    score_types = ["original", "recalibrated"]

    # Calculate overall scores using pandas' dot product
    for score_type in score_types:
        columns = [f"{score_type}_{skill}" for skill in weights.keys()]
        df[score_type + "_overall_score"] = (
            df[columns].values.dot(pd.Series(weights).values) / 100.0
        )

    return df
