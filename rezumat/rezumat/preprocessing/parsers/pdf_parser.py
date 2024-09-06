# read all files
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Union

from pypdf import PdfReader
from tqdm import tqdm


def parse_pdf(file_path):
    reader = PdfReader(file_path)
    return [page.extract_text() for page in reader.pages]


def process_pdfs(pdf_path: Union[str, Path]) -> List[str]:
    cv_data = []
    total_pdfs = len(list(Path(pdf_path).glob("*.pdf")))
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_file = {
            executor.submit(parse_pdf, file): file for file in pdf_path.glob("*.pdf")
        }

        for future in tqdm(
            as_completed(future_to_file), total=total_pdfs, desc="Parsing PDFs"
        ):
            cv_data.append(future.result())

    # Join multi-page CVs into a single string
    return [" ".join(cv) if len(cv) > 1 else cv[0] for cv in cv_data]
