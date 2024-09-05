from __future__ import annotations

from typing import List, Union

import gradio as gr
import pandas as pd

from .config import config
from .preprocessing.input_data_processing import process_input
from .utils.helper import format_job_description_analysis, set_and_verify_api_key
from .utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------
# gradio interface
# ------------------------------


def create_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown(f"# {config.TITLE}")

        # add a state to store the eval_results
        eval_results = gr.State()
        api_key_status = gr.State()

        # INITIAL VIEW
        with gr.Group() as initial_view:
            with gr.Row():
                with gr.Column(scale=2):

                    ## sidebar (20% of the screen)
                    # model selection section
                    api_key = gr.Textbox(label="API Key", type="password")
                    interface = gr.Dropdown(
                        ["Groq", "OpenAI", "Anthropic"], label="Interface", value="Groq"
                    )
                    model = gr.Dropdown(
                        ["llama3-70b-8192", "gpt-3.5-turbo", "gpt-4"],
                        label="Model",
                        value="llama3-70b-8192",
                    )

                    # weights section
                    gr.Markdown("### Weights (Total must be 100)")
                    technical_skills = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=60,
                        step=1,
                        label="Technical Skills",
                    )
                    soft_skills = gr.Slider(
                        minimum=0, maximum=100, value=10, step=1, label="Soft Skills"
                    )
                    experience = gr.Slider(
                        minimum=0, maximum=100, value=20, step=1, label="Experience"
                    )
                    education = gr.Slider(
                        minimum=0, maximum=100, value=10, step=1, label="Education"
                    )

                with gr.Column(scale=8):

                    ## Main content (80% of the screen)
                    # job description section
                    jd_text_input = gr.TextArea(
                        label="Job Description",
                        lines=12,
                        info="Paste the job description here",
                    )
                    input_type = gr.Radio(
                        ["Text", "File"],
                        label="Select Resume Type",
                        value="Text",
                        info="Select the type of input",
                    )

                    # resume section (text or file)
                    with gr.Row():
                        additional_text = gr.Textbox(
                            label="Resume",
                            lines=5,
                            visible=True,
                            info="Paste the resume here",
                        )
                        file_upload = gr.File(
                            label="Upload File",
                            file_count="multiple",
                            file_types=["pdf"],
                            visible=False,
                        )

                    with gr.Row():
                        submit_btn = gr.Button("Evaluate")
                        reset_btn = gr.Button("Reset")

        # RESULTS VIEW (INITIALLY HIDDEN)
        with gr.Group(visible=False) as results_view:

            with gr.Row():
                # applicant summary
                total_applicants = gr.Number(label="Total Applicants")
                yes_count = gr.Number(label="Yes")
                no_count = gr.Number(label="No")
                kiv_count = gr.Number(label="KIV")

            with gr.Row(equal_height=True):
                with gr.Column():
                    # suitability filter
                    suitability_filter = gr.Radio(
                        ["All", "Yes", "No", "KIV"],
                        label="filter by suitability",
                        value="All",
                    )
                with gr.Column():
                    # filter candidates by cv_id
                    top_candidates = gr.Dropdown(
                        label="Top Candidates",
                    )

            with gr.Row():
                with gr.Column():
                    with gr.Tabs():
                        # job description & analysis
                        with gr.TabItem("Job Description"):
                            jd_display = gr.TextArea(
                                label="Job Description", interactive=False
                            )
                        with gr.TabItem("JD Analysis"):
                            job_analysis_display = gr.Markdown(label="Job Analysis")

                with gr.Column():
                    # candidate details
                    gr.Markdown("## Candidate Details")
                    score_comparison = gr.Markdown(label="Score Comparison")
                    cv_display = gr.TextArea(label="Selected CV", interactive=False)
                    strengths = gr.TextArea(label="Strengths", interactive=False)
                    concerns = gr.TextArea(label="Concerns", interactive=False)

            back_btn = gr.Button("Back")

        # Event handlers: update input type (text or file)
        def update_input_type(choice):
            if choice == "Text":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        # Event handlers: reset interface
        def reset_interface():
            return [
                "",  # jd_text_input
                "",  # additional_text
                None,  # file_upload
                "Text",  # input_type
                "",  # api_key
                "Groq",  # interface
                "llama3-70b-8192",  # model
                60,  # technical_skills
                10,  # soft_skills
                20,  # experience
                10,  # education
            ]

        # Event handlers: process results
        def process_results(results_df: Union[pd.DataFrame, List[dict]]):

            logger.info("Processing results...")

            if results_df is None or results_df.empty:
                logger.warning("DataFrame is None or empty")
                return [
                    gr.update(visible=True),  # Keep initial view visible
                    gr.update(visible=False),  # Keep results view hidden
                    None,
                    None,
                    None,
                    None,  # Total, Yes, No, KIV counts
                    None,  # Top candidates dropdown
                    "",  # Job description
                    None,  # Eval results state
                    "Error: No results received from process_input function",  # Debug output
                ]

            if not isinstance(results_df, pd.DataFrame):
                logger.error(f"Expected a pandas DataFrame, but got {type(results_df)}")
                return [
                    gr.update(visible=True),  # Keep initial view visible
                    gr.update(visible=False),  # Keep results view hidden
                    None,
                    None,
                    None,
                    None,  # Total, Yes, No, KIV counts
                    None,  # Top candidates dropdown
                    "",  # Job description
                    None,  # Eval results state
                    "Error: DataFrame expected",  # Debug output
                ]

            try:
                total = len(results_df)
                yes = len(results_df[results_df["suitability"] == "yes"])
                no = len(results_df[results_df["suitability"] == "no"])
                kiv = len(results_df[results_df["suitability"] == "kiv"])

                top_candidates = results_df.sort_values(
                    by="recalibrated_overall_score", ascending=False
                ).head(5)
                top_candidates_list = top_candidates["cv_id"].tolist()

                job_description = (
                    results_df["job_text"].iloc[0] if not results_df.empty else ""
                )
                job_analysis = (
                    results_df["job_analysis"].iloc[0] if not results_df.empty else ""
                )
                job_analysis_markdown = format_job_description_analysis(job_analysis)

                return (
                    gr.update(visible=False),  # hide initial view
                    gr.update(visible=True),  # show results
                    total,
                    yes,
                    no,
                    kiv,
                    gr.Dropdown(
                        choices=top_candidates_list,
                        value=top_candidates_list[0] if top_candidates_list else None,
                    ),
                    job_description,
                    job_analysis_markdown,
                    results_df,  # store full results in state
                )
            except Exception as e:
                error_msg = f"Error processing results: {str(e)}"
                logger.error(error_msg)
                return [
                    gr.update(visible=True),  # Keep initial view visible
                    gr.update(visible=False),  # Keep results view hidden
                    None,
                    None,
                    None,
                    None,  # Total, Yes, No, KIV counts
                    None,  # Top candidates dropdown
                    "",  # Job description
                    None,  # Eval results state
                    error_msg,  # Debug output
                ]

        def update_candidate_list(suitability, results_df):
            if results_df is None or results_df.empty:
                return gr.Dropdown(choices=[], value=None)

            if suitability != "All":
                filtered_df = results_df[
                    results_df["suitability"] == suitability.lower()
                ]

                if filtered_df.empty:
                    filtered_df = results_df
                    gr.Warning(
                        "No candidates found for the selected suitability filter. Showing all candidates."
                    )
            else:
                filtered_df = results_df

            candidates = filtered_df.sort_values(
                by="recalibrated_overall_score", ascending=False
            )
            candidate_list = candidates["cv_id"].tolist()

            return gr.Dropdown(
                choices=candidate_list,
                value=candidate_list[0] if candidate_list else None,
            )

        def display_candidate_info(cv_id, results_df):

            if results_df is None or results_df.empty:
                return (
                    gr.update(value="No candidates found"),
                    gr.update(value=""),
                    gr.update(value=""),
                )
            matching_candidates = results_df[results_df["cv_id"] == cv_id]

            if matching_candidates.empty:
                logger.warning(f"No matching candidates found for cv_id: {cv_id}")
                return "No matching candidates found", "", ""

            candidate_info = matching_candidates.iloc[0]

            return (
                candidate_info["cv_text"],
                candidate_info.get("strengths", "No strengths information available"),
                candidate_info.get("concerns", "No concerns information available"),
            )

        def display_score_comparison(cv_id, results_df):
            if results_df is None or results_df.empty:
                return gr.update(value="No score comparison available"), ""

            candidate_info = results_df[results_df["cv_id"] == cv_id].iloc[0]

            markdown_text = """
                | | original_score | recalibrated_score |
                |------------------|----------------|--------------------|
                | technical_skills | {:.2f} | {:.2f} |
                | soft_skills | {:.2f} | {:.2f} |
                | experience | {:.2f} | {:.2f} |
                | education | {:.2f} | {:.2f} |
                | overall | {:.2f} | {:.2f} |
                """.format(
                candidate_info["original_technical_skills"],
                candidate_info["original_overall_score"],
                candidate_info["original_soft_skills"],
                candidate_info["recalibrated_soft_skills"],
                candidate_info["original_experience"],
                candidate_info["recalibrated_experience"],
                candidate_info["original_education"],
                candidate_info["recalibrated_education"],
                candidate_info["original_overall_score"],
                candidate_info["recalibrated_overall_score"],
            )

            return markdown_text

        # Event handlers
        input_type.change(
            update_input_type,
            inputs=[input_type],
            outputs=[additional_text, file_upload],
        )

        submit_btn.click(
            fn=set_and_verify_api_key,
            inputs=[api_key, interface],
            outputs=api_key_status,
        ).success(
            fn=process_input,
            inputs=[
                jd_text_input,
                additional_text,
                file_upload,
                input_type,
                api_key,
                interface,
                model,
                technical_skills,
                soft_skills,
                experience,
                education,
            ],
            outputs=eval_results,
        ).then(
            fn=process_results,
            inputs=[eval_results],
            outputs=[
                initial_view,
                results_view,
                total_applicants,
                yes_count,
                no_count,
                kiv_count,
                top_candidates,
                jd_display,
                job_analysis_display,
                eval_results,
            ],
        )

        reset_btn.click(
            fn=reset_interface,
            inputs=[],
            outputs=[
                jd_text_input,
                additional_text,
                file_upload,
                input_type,
                api_key,
                interface,
                model,
                technical_skills,
                soft_skills,
                experience,
                education,
            ],
        )

        suitability_filter.change(
            fn=update_candidate_list,
            inputs=[suitability_filter, eval_results],
            outputs=[top_candidates],
        )

        top_candidates.change(
            fn=display_candidate_info,
            inputs=[top_candidates, eval_results],
            outputs=[cv_display, strengths, concerns],
        ).then(
            fn=display_score_comparison,
            inputs=[top_candidates, eval_results],
            outputs=[score_comparison],
        )

        back_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            inputs=[],
            outputs=[initial_view, results_view],
        )

    return demo
