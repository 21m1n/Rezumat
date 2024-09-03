# read all files 
from concurrent.futures import ThreadPoolExecutor, as_completed
import os 
from tqdm import tqdm
from pypdf import PdfReader
from pathlib import Path

pdf_path = Path("data/resumes")
os.makedirs(pdf_path, exist_ok=True)

num_threads = os.cpu_count()
total_pdfs = len(list(pdf_path.glob("*.pdf")))
cv_data = []

def process_pdf(file_path):
    reader = PdfReader(file_path)
    return [page.extract_text() for page in reader.pages]



with ThreadPoolExecutor(max_workers=num_threads) as executor:
    future_to_file = {executor.submit(process_pdf, file): file for file in pdf_path.glob("*.pdf")}
    
    for future in tqdm(as_completed(future_to_file), total=total_pdfs):
        cv_data.append(future.result())

# Join multi-page CVs into a single string
cv_data = [" ".join(cv) if len(cv) > 1 else cv[0] for cv in cv_data]