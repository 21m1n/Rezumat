import gradio as gr
import pandas as pd
import json
from pathlib import Path

# Load the data
main_data_path = Path("../notebooks/")
final_verdict_path = Path.joinpath(main_data_path, "output/disagreement_gpt4o_llama3_70b.csv")
final_verdict = pd.read_csv(final_verdict_path)

# Global variable to keep track of the current index
current_index = 0

import json
from tabulate import tabulate

def format_json_to_markdown(json_data):
    data = json.loads(json_data)
    
    markdown = "# Assessment:\n\n"
    markdown += f"- **Suitability:** {data['assessment']['suitability']}\n"
    markdown += "  - **Reasons:**\n"
    for reason in data['assessment']['reasons']:
        markdown += f"    - {reason}\n"
    markdown += "  - **Missing Skills:**\n"
    for skill in data['assessment']['missing_skills']:
        markdown += f"    - {skill}\n"
    markdown += "  - **Potential Concerns:**\n"
    for concern in data['assessment']['potential_concerns']:
        markdown += f"    - {concern}\n"
    markdown += "  - **Strengths:**\n"
    for strength in data['assessment']['strengths']:
        markdown += f"    - {strength}\n"

    markdown += "\n## Scores:\n\n"
    scores = [
        ["Category", "Original Score", "Recalibrated Score"],
        ["Technical Skills", data['resume_evaluation']['original_scores']['technical_skills'], data['recalibrated_scores']['technical_skills']],
        ["Soft Skills", data['resume_evaluation']['original_scores']['soft_skills'], data['recalibrated_scores']['soft_skills']],
        ["Experience", data['resume_evaluation']['original_scores']['experience'], data['recalibrated_scores']['experience']],
        ["Education", data['resume_evaluation']['original_scores']['education'], data['recalibrated_scores']['education']]
    ]
    markdown += tabulate(scores, headers="firstrow", tablefmt="pipe") + "\n\n"

    markdown += "**Reasons:**\n"
    markdown += "- **Explicit Missing Skills:**\n"
    for skill in data['resume_evaluation']['missing_skills']:
        markdown += f"  - {skill}\n"
    markdown += "- **Inferred Experiences:**\n"
    for skill, inference in data['deeper_analysis']['inferred_experiences'].items():
        markdown += f"  - {skill}: {inference}\n"

    markdown += "\n## Job Description Analysis:\n"
    markdown += "- **Essential Tech Skills:**\n"
    for skill in data['job_description_analysis']['technical_skills']['essential']:
        markdown += f"  - {skill}\n"
    markdown += "- **Advantageous Skills:**\n"
    for skill in data['job_description_analysis']['technical_skills']['advantageous']:
        markdown += f"  - {skill}\n"
    markdown += "- **Soft Skills:**\n"
    for skill in data['job_description_analysis']['soft_skills']:
        markdown += f"  - {skill}\n"
    markdown += f"- **Level of Experience:** {data['job_description_analysis']['level_of_exp']}\n"
    markdown += "- **Education:**\n"
    for edu in data['job_description_analysis']['education']:
        markdown += f"  - {edu}\n"

    return markdown

def load_llama3_reasoning(job_id, cv_id):
    json_path = Path.joinpath(main_data_path, f"output_20240831_0124/{job_id}_{cv_id}_groq.json")
    if json_path.exists():
      try:  
        with open(json_path, "r") as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
      except Exception as e:
        return f"Error loading reasoning: {e}"
    return "Reasoning not available"

def load_gpt4o_reasoning(job_id, cv_id):
    json_path = Path.joinpath(main_data_path, f"output_20240831_1144/{job_id}_{cv_id}_gpt4o.json")
    if json_path.exists():
      try:
        with open(json_path, "r") as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
      except Exception as e:
        return f"Error loading reasoning: {e}"
    return "Reasoning not available"

def load_entry(index):
    entry = final_verdict.iloc[index]
    llama3_reasoning = load_llama3_reasoning(entry['job_id'], entry['cv_id'])
    gpt4o_reasoning = load_gpt4o_reasoning(entry['job_id'], entry['cv_id'])
    return (
        entry['job_title'],
        entry['job_description'],
        entry['cv_category'],
        entry['cv'],
        f"GPT-4o: {entry['gpt4o']}\nLlama3-70b: {entry['llama3_70b']}\n\nAnthropic: {entry['anthropic']}\nGPT-3.5-turbo: {entry['gpt']}\nLlama3: {entry['llama3']}",
        llama3_reasoning,
        gpt4o_reasoning
    )

def update_decision(decision):
    global current_index
    final_verdict.loc[current_index, 'user_decision'] = decision
    final_verdict.to_csv(Path.joinpath(main_data_path, "output/disagreement_gpt4o_llama3_70b_with_user_decision.csv"), index=False)
    current_index += 1
    return load_entry(current_index)

def shortlist():
    return update_decision("Shortlist")

def reject():
    return update_decision("Reject")

def kiv():
    return update_decision("KIV")

def load_first_entry():
    return load_entry(0)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            job_title = gr.Textbox(label="Job Title")
            job_description = gr.Textbox(label="Job Description", lines=10)
        with gr.Column():
            cv_category = gr.Textbox(label="CV Category")
            cv = gr.Textbox(label="Resume (CV)", lines=10)
            
    model_verdicts = gr.Textbox(label="Model Verdicts", lines=3)
    
    with gr.Row():
      with gr.Column():
        llama3_reasoning = gr.Code(label="Llama3 Reasoning", language="json")
      with gr.Column():
        gpt4o_reasoning = gr.Code(label="GPT4o Reasoning", language="json")
    
    with gr.Row():
        shortlist_btn = gr.Button("Shortlist", variant="primary")
        reject_btn = gr.Button("Reject", variant="stop")
        kiv_btn = gr.Button("KIV", variant="secondary")
    
    shortlist_btn.click(shortlist, outputs=[job_title, job_description, cv_category, cv, model_verdicts, llama3_reasoning, gpt4o_reasoning])
    reject_btn.click(reject, outputs=[job_title, job_description, cv_category, cv, model_verdicts, llama3_reasoning, gpt4o_reasoning])
    kiv_btn.click(kiv, outputs=[job_title, job_description, cv_category, cv, model_verdicts, llama3_reasoning, gpt4o_reasoning])
    
    demo.load(load_first_entry, outputs=[job_title, job_description, cv_category, cv, model_verdicts, llama3_reasoning, gpt4o_reasoning])

demo.launch()