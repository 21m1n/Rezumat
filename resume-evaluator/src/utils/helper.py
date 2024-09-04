import json

def format_job_description_analysis(json_data):
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
    except json.JSONDecodeError:
        return "Error: Invalid JSON data provided."

    markdown = "# Job Description Analysis\n\n"

    # Technical Skills
    markdown += "## Technical Skills\n\n"
    markdown += "### Essential\n"
    for skill in data['technical_skills'].get('essential', []):
        markdown += f"- {skill}\n"
    if not data['technical_skills'].get('essential'):
        markdown += "- No essential technical skills listed\n"

    markdown += "\n### Advantageous\n"
    for skill in data['technical_skills'].get('advantageous', []):
        markdown += f"- {skill}\n"
    if not data['technical_skills'].get('advantageous'):
        markdown += "- No advantageous technical skills listed\n"

    # Soft Skills
    markdown += "\n## Soft Skills\n"
    for skill in data.get('soft_skills', []):
        markdown += f"- {skill}\n"
    if not data.get('soft_skills'):
        markdown += "- No soft skills listed\n"

    # Level of Experience
    markdown += f"\n## Level of Experience\n"
    markdown += f"- {data.get('level_of_exp', 'Not specified')}\n"

    # Education
    markdown += "\n## Education\n"
    for edu in data.get('education', []):
        markdown += f"- {edu}\n"
    if not data.get('education'):
        markdown += "- No specific education requirements listed\n"

    return markdown