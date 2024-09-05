import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import gradio as gr

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


def set_api_key(api_key: str, model_text: str) -> str:
    """set the api key for the model"""

    env_var_map = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": "OLLAMA_API_KEY",
    }
    env_var = env_var_map.get(model_text.lower())

    if env_var:
        os.environ[env_var] = api_key
        logger.info(f"API key set successfully for {model_text}")
    else:
        logger.error(f"Invalid model text: {model_text}")


def format_job_description_analysis(json_data):
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
    except json.JSONDecodeError:
        return "Error: Invalid JSON data provided."

    markdown = "# Job Description Analysis\n\n"

    # Technical Skills
    markdown += "## Technical Skills\n\n"
    markdown += "### Essential\n"
    for skill in data["technical_skills"].get("essential", []):
        markdown += f"- {skill}\n"
    if not data["technical_skills"].get("essential"):
        markdown += "- No essential technical skills listed\n"

    markdown += "\n### Advantageous\n"
    for skill in data["technical_skills"].get("advantageous", []):
        markdown += f"- {skill}\n"
    if not data["technical_skills"].get("advantageous"):
        markdown += "- No advantageous technical skills listed\n"

    # Soft Skills
    markdown += "\n## Soft Skills\n"
    for skill in data.get("soft_skills", []):
        markdown += f"- {skill}\n"
    if not data.get("soft_skills"):
        markdown += "- No soft skills listed\n"

    # Level of Experience
    markdown += f"\n## Level of Experience\n"
    markdown += f"- {data.get('level_of_exp', 'Not specified')}\n"

    # Education
    markdown += "\n## Education\n"
    for edu in data.get("education", []):
        markdown += f"- {edu}\n"
    if not data.get("education"):
        markdown += "- No specific education requirements listed\n"

    return markdown


def save_upload_file(file) -> None:
    """save the file uploaded by the user to the pdf upload folder PDF_UPLOAD_FOLDER"""
    if not os.path.exists(config.PDF_UPLOAD_FOLDER):
        os.makedirs(config.PDF_UPLOAD_FOLDER, exist_ok=True)
    shutil.copy(file, config.PDF_UPLOAD_FOLDER)
    gr.Info(f"file is saved to {config.PDF_UPLOAD_FOLDER}{file.name.split('/')[-1]}")


# [TODO] to remove?
def read_job_data() -> List[Tuple[str, dict]]:
    """Read job data from JOBS_OUTPUT_DIR"""
    job_data = []
    for file in Path(config.JOBS_OUTPUT_DIR).glob("*.json"):
        job_id = file.stem.split("_")[0]
        with open(file, "r") as f:
            job_analysis = json.load(f)
            job_data.append((job_id, job_analysis))
    return job_data
