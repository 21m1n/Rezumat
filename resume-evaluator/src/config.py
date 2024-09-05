import logging
import shutil
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    TITLE: str = "✏️ Resume Evaluator"
    BASE_DIR: Path = Path("./resume-evaluator").resolve()
    PDF_UPLOAD_FOLDER: Path = BASE_DIR / "data/input/pdf"
    OUTPUT_DIR: Path = BASE_DIR / "data/output"
    CSV_OUTPUT_DIR: Path = OUTPUT_DIR / "csv"
    JOBS_OUTPUT_DIR: Path = OUTPUT_DIR / "jobs"
    CV_OUTPUT_DIR: Path = OUTPUT_DIR / "cv"
    ENV_PATH: Path = BASE_DIR / ".env"
    LOG_FILE: Path = BASE_DIR / "logs/evaluation_log.txt"
    LOG_LEVEL: str = "INFO"
    SLEEP_TIME: float = 2.1  # add a small delay to avoid rate limiting

    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 8192

    GROQ_URL: str = "https://api.groq.com/openai/v1/models"
    OPENAI_URL: str = "https://api.openai.com/v1/models"
    ANTHROPIC_URL: str = "https://api.anthropic.com/v1/models"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        arbitrary_types_allowed=True,
        allow_extra="allow",
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories, removing existing ones if they exist."""
        directories = [
            self.PDF_UPLOAD_FOLDER,
            self.JOBS_OUTPUT_DIR,
            self.CV_OUTPUT_DIR,
            self.CSV_OUTPUT_DIR,
        ]
        for directory in directories:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=self.LOG_FILE,
            filemode="a",
        )


# global instance of Config
config = Config()
config.setup_logging()
