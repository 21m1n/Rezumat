from pathlib import Path

import pytest

from rezumat.preprocessing.parsers.pdf_parser import parse_pdf


def test_parse_pdf_valid_file():
    pdf_path = "tests/fixtures/sample_resume.pdf"
    result = parse_pdf(pdf_path)
    assert isinstance(result, list)
    assert result != []


def test_parse_pdf_invalid_file():
    with pytest.raises(FileNotFoundError):
        parse_pdf("tests/fixtures/non_existent_file.pdf")


def test_parse_pdf_empty_file():
    empty_pdf = Path("tests/fixtures/empty_pdf.pdf")
    empty_pdf.touch()
    result = parse_pdf(empty_pdf)
    assert result == ""
