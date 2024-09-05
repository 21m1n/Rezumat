import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Union
from uuid import uuid4

import pandas as pd
from langchain_core.runnables.base import RunnableSequence
from tqdm import tqdm

from ..config import config
from ..evaluators.two_stage_evaluators import two_stage_eval_cv, two_stage_eval_jd
from ..utils.logger import get_logger

logger = get_logger(__name__)


def process_all_jobs(
    model_tuples: List[Tuple[str, RunnableSequence]],
    job_text: Union[str, List[str]],
    output_dir: Union[str, Path],
):

    # create the job tuple which consists of job_id and job_text
    if isinstance(job_text, List):
        job_tuples = [(str(uuid4()), jt) for jt in job_text]
    else:
        job_tuples = [(str(uuid4()), job_text)]

    pd.DataFrame(job_tuples, columns=["job_id", "job_text"]).to_csv(
        os.path.join(config.CSV_OUTPUT_DIR, "job_tuples.csv"), index=False
    )
    logger.info(f"saved job tuple")

    # [TODO] change to the number of cores in the machine (add to config)
    with ThreadPoolExecutor(max_workers=1) as executor:

        futures = []

        for job_tuple in job_tuples:
            futures.append(
                executor.submit(two_stage_eval_jd, model_tuples, job_tuple, output_dir)
            )

        for future in tqdm(
            as_completed(futures), total=len(job_tuples), desc="Processing all jobs"
        ):
            try:
                future = future.result()
            except Exception as e:
                print(f"Processing Error: {e}")


def process_all_pairs(
    model_tuples: List[Tuple[str, RunnableSequence]],
    job_data: List[Tuple[str, str]],
    cv_data: List[Tuple[str, str]],
    output_dir: str,
):

    total_pairs = len(job_data) * len(cv_data)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for job in job_data:
            for cv in cv_data:
                futures.append(
                    executor.submit(
                        two_stage_eval_cv, model_tuples, job, cv, output_dir
                    )
                )

        for future in tqdm(
            as_completed(futures), total=total_pairs, desc="Processing job-cv pairs"
        ):
            try:
                future = future.result()
            except Exception as e:
                print(f"Error: {e}")
