import pytest
from unittest.mock import Mock
from langchain_core.runnables.base import RunnableSequence


# creating mock objects for the test
@pytest.fixture
def mock_model_tuple():
    mock_grader = Mock(spec=RunnableSequence)
    mock_grader.invoke.return_value = {
        "technical_skills": {
            "essential": ["Python", "JavaScript"],
            "advantageous": ["Docker", "AWS"],
        },
        "soft_skills": ["Communication", "Teamwork"],
        "level_of_exp": "Mid-level",
        "education": ["Bachelor's in Computer Science"],
    }
    return ("model1", mock_grader)


@pytest.fixture
def mock_job_tuple():
    return (
        "123456",
        "software engineer with 3 years of experience specializing in python and machine learning",
    )


@pytest.fixture
def mock_output_dir():
    return "tests/fixtures"


@pytest.fixture
def mock_cv_tuple():
    return ("456789", "Experienced Python developer with 7 years in the industry")
