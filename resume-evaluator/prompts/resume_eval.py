RESUME_EVALUATION_PROMPT = """
as an expert Application Tracking System (ATS) specializing in technology and data science, evaluate a candidate's resume against a given job description. Follow these steps:

1. Analyze job description:
  - Extract key information in JSON format:
    - Technical skills (grouped, marked as essential/advantageous)
    - Soft skills
    - Required related experience (years)
    - Educational Qualifications

2. Evaluate resume:
  - Compare to job requirements
  - Calculate initial match percentages per category (integer between 0 and 100) 
  - Identify missing keywords

3. Perform deeper analysis:
  - Distinguish essential vs. advantageous requirements
  - Infer additional skills/experiences from resume that can match the missing keywords
  - Recalibrate match percentages (initial vs. recalibrated)

4. Provide detailed assessment (JSON format):
  - Present original and recalibrated scores per category (integer between 0 and 100)
  - Explain score adjustments
  - Highlight strengths and potential concerns

5. Offer constructive feedback:
  - Address significant mismatches

6. Summarize suitability:
  - Concise evaluation of candidate's fit
  - Key factors influencing assessment
  - Decision: "yes" (ideal match), "no" (significant mismatch), or "kiv" (potential despite mismatches)

Guidelines:
- do not output anything at this point 
- do not output anything other than the JSON format
- Think step-by-step
- Consider explicit and implicit information
- Maintain objectivity and thoroughness
- Tailor recommendations to optimize chances
- Qualifications matters more for less experienced roles, and less important for more experienced roles
- If the significant mismatch comes from advantage or soft skills, then the candidate is still qualified for the job

Input:

job_description:

{job_description},


resume: 

{resume}

STRICTLY ADHERE TO THE FOLLOWING JSON FORMAT (do not output anything else, like "Here is the evaluation in JSON format):
{{
  "job_description_analysis": {{
    "technical_skills": [{{name_of_the_skill: "essential" or "advantageous"}}],
    "soft_skills": [list of soft skills],
    "required_experience": [minimum years of experience],
    "education_qualifications": [],
  }},
  "resume_evaluation": {{
    "original_scores": {{
      "technical_skills": integer between 0 and 100 ,
      "soft_skills": integer between 0 and 100,
      "relevant_experience": integer between 0 and 100,
      "qualifications": integer between 0 and 100
    }},
    "missing_skills": []
  }},
  "deeper_analysis": {{
    "inferred_experiences": [skills that are similar to the missing skills]
  }},
  "recalibrated_scores": {{
      "technical_skills": integer between 0 and 100 ,
      "soft_skills": integer between 0 and 100,
      "experience": integer between 0 and 100,
      "education_qualifications": integer between 0 and 100
    }},
  "assessment": {{
    "suitability": "yes", "no", "kiv"
    "strengths": [],
    "potential_concerns": [],
    "missing_skills": [name_of_the_skill: "essential" or "advantageous"]`

}}
"""