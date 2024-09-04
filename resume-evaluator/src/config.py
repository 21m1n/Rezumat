import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()  # This loads the .env file
        
        self.BASE_DIR = Path("./resume-evaluator").resolve()
        self.PDF_UPLOAD_FOLDER = self.BASE_DIR / "data/input/pdf"
        self.OUTPUT_DIR = self.BASE_DIR / "data/output"
        self.CSV_OUTPUT_DIR = self.OUTPUT_DIR / "csv"
        self.JOBS_OUTPUT_DIR = self.OUTPUT_DIR / "jobs"
        self.CV_OUTPUT_DIR = self.OUTPUT_DIR / "cv"
        self.ENV_PATH = self.BASE_DIR / ".env"
        self.LOG_FILE = self.BASE_DIR / "logs/evaluation_log.txt"
        
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", 8192))
        
        # API keys and other sensitive data
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        
        # Initialize directories
        self.setup_directories()
    
    def __getattr__(self, name):
        return getattr(self, name)
    
    def setup_directories(self):
        """Create necessary directories, removing existing ones if they exist."""
        directories = [self.PDF_UPLOAD_FOLDER, self.JOBS_OUTPUT_DIR, self.CV_OUTPUT_DIR, self.CSV_OUTPUT_DIR]
        for directory in directories:
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

config = Config()