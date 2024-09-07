import logging
import shutil
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    TITLE: str = "✏️ Resume Evaluator"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
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
        # [TODO] come up with a better way to handle this
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

    def setup_logging(self, log_file: Optional[Path] = None):
        """Setup logging configuration.

        Args:
            log_file: Path to the log file. If not provided, the default log file path is used.
        """
        log_file = log_file or self.LOG_FILE
        logger = logging.getLogger("[Rezumat]")
        logger.setLevel(logging.DEBUG)

        # console handler
        ch = logging.StreamHandler()
        ch.setLevel(self.LOG_LEVEL)

        # rotating file handler (10MB per file, max 5 files)
        fh = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 10, backupCount=5)
        fh.setLevel(self.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(ch)
        logger.addHandler(fh)

        logger.addHandler(ch)
        logger.addHandler(fh)

        return logger

    def update_python_path(self):
        """Update the Python path to include the project root."""
        project_root = self.BASE_DIR.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))


# global instance of Config
config = Config()
config.update_python_path()

logger = config.setup_logging()
