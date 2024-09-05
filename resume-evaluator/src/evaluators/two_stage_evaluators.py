import json
import os
import time
from typing import List, Tuple, Union

import pandas as pd
from langchain_core.runnables import RunnableSequence

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


def two_stage_eval_jd(
    model_tuples: List[Tuple[str, RunnableSequence]],
    job_tuple: Tuple[str, str],
    output_dir: str,
) -> Union[pd.DataFrame, None]:
    model_results = {}

    logger.info(f"Start two-stage evaluation for job description.")

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
                logger.info(f"Saved {model_name} result for job_id: {job_id}")
            time.sleep(config.SLEEP_TIME)  # Add a small delay to avoid rate limiting

        except Exception as e:
            error_msg = f"Error with {model_name} for job_id: {job_id}. Error: {str(e)}"
            logger.error(error_msg)
            print(error_msg)

    if not model_results:
        error_msg = f"All models failed for job_id: {job_id}."
        logger.error(error_msg)
        print(error_msg)
        return None


def two_stage_eval_cv(
    model_tuples: List[Tuple[str, RunnableSequence]],
    job_tuple: Tuple[str, str],
    cv_tuple: Tuple[str, str],
    output_dir: str,
) -> Union[pd.DataFrame, None]:

    logger.info(f"Start two-stage evaluation for job description.")

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
            logger.error(error_msg)
            print(error_msg)

    if not model_results:
        error_msg = f"All models failed for job_id: {job_id}."
        logger.error(error_msg)
        print(error_msg)
        return None
