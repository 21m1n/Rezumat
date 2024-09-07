from unittest.mock import Mock, mock_open, patch

from langchain_core.runnables import RunnableSequence

from rezumat.evaluators.two_stage_evaluators import two_stage_eval_jd, two_stage_eval_cv
from rezumat.config import config


def test_successful_evaluation(mock_model_tuple, mock_job_tuple, mock_output_dir):
    """check if two_stage_eval_jd() correctly opens a file and writes the output as json."""
    # mock the open() function to return a file object
    m = mock_open()
    with patch("builtins.open", m), patch("json.dump") as mock_json_dump:
        two_stage_eval_jd(mock_model_tuple, mock_job_tuple, mock_output_dir)

    # ensure that open() was called exactly once with the correct arguments
    m.assert_called_once_with(f"{mock_output_dir}/123456_model1.json", "w")

    # checks that json.dump() was called once with the correct arguments
    mock_json_dump.assert_called_once_with(
        mock_model_tuple[1].invoke.return_value, m(), indent=4
    )


def test_all_models_fail(mock_job_tuple, mock_output_dir):
    """check if two_stage_eval_jd() correctly handles the case where all models fail."""
    mock_grader = Mock(spec=RunnableSequence)

    # configure the mock grader to raise an exception when invoked
    mock_grader.invoke.side_effect = Exception("Model failed")

    model_tuples = [("failed_model", mock_grader)]

    # mock the print function to prevent actual output
    with patch("builtins.print"):
        result = two_stage_eval_jd(model_tuples, mock_job_tuple, mock_output_dir)

    assert result is None


@patch("time.sleep")
def test_rate_limiting(mock_sleep, mock_model_tuple, mock_job_tuple, mock_output_dir):
    model_tuples = [mock_model_tuple]

    with patch("json.dump"), patch("builtins.open", mock_open()):
        two_stage_eval_jd(model_tuples, mock_job_tuple, mock_output_dir)

    mock_sleep.assert_called_once_with(config.SLEEP_TIME)


def test_two_stage_eval_cv_success(
    mock_model_tuple, mock_job_tuple, mock_cv_tuple, mock_output_dir
):
    """Check if two_stage_eval_cv() correctly opens a file and writes the output as json."""

    m = mock_open()
    with patch("builtins.open", m), patch("json.dump") as mock_json_dump:
        two_stage_eval_cv(
            [mock_model_tuple], mock_job_tuple, mock_cv_tuple, mock_output_dir
        )

    # Ensure that open() was called exactly once with the correct arguments
    expected_filename = f"{mock_output_dir}/123456_456789_model1.json"
    m.assert_called_once_with(expected_filename, "w")

    # Check that json.dump() was called once with the correct arguments
    mock_json_dump.assert_called_once_with(
        mock_model_tuple[1].invoke.return_value, m(), indent=4
    )

    # Optionally, verify that the model's invoke method was called with the correct arguments
    mock_model_tuple[1].invoke.assert_called_once_with(
        {"job_requirements": mock_job_tuple[1], "resume": mock_cv_tuple[1]}
    )
