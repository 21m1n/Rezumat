RESUME_EVALUATION_PROMPT = """
as an expert Application Tracking System (ATS) specializing in technology and data science, evaluate a candidate's resume against a given job description. Follow these steps:

1. Analyze job description:
  - Extract key information from the job description only:
    - technical_skills: dictionary, e.g. {{"essential": ["SQL", "Python], "advantageous": ["AWS"]}}
    - soft_skills: list of strings, e.g. ["communication", "teamwork"]
    - level_of_exp: "junior", "mid-level", "senior", or "executive"
    - education: list of strings e.g. ["Master's degree in Computer Science"]

2. Evaluate resume:
  - extract the key information from the resume only
  - compare to the information extracted from the job description
  - calculate the percentage of match (integer between 0 and 100) per category, e.g. [90]
  - Identify ONLY the essential skills listed in the job description that are missing from the resume, e.g. ["SQL"]

3. Perform deeper analysis:
  - review the resume again, see if the missing skills can be inferred from the resume, e.g. "SQL" can be inferred from "database", 
    "deep learning" can be inferred from "tensorflow" or "pytorch"
  - if so, indicate the inferred skills, inferred_skills, dictionary, e.g. ["SQL": "inferred from database"]
  - Recalibrate match percentages, recalibrated_scores: list of integers between 0 and 100 

4. Provide consolidate the findings and present the final assessment in JSON format:
  - job_description_analysis: dictionary
    - technical_skills: dictionary
    - soft_skills: list of strings, e.g. ["communication", "teamwork"]
    - level_of_exp: "junior", "mid-level", "senior", or "executive"
    - education: list of strings e.g. ["Master's degree in Computer Science"]
  - resume_evaluation: dictionary, 
    - original_scores: dictionary, e.g. {{"technical_skills": 60, "soft_skills": 80, "experience": 70, "education": 90]}}
    - missing_skills: list of strings, e.g. ["SQL", "Python"]
  - deeper_analysis: dictionary
    - inferred_experiences: dictionary
  - recalibrated_scores: dictionary
  - assessment: dictionary
    - suitability: "yes", "no", or "kiv"
    - strengths: list of strings
    - potential_concerns: list of strings
    - missing_skills: list of strings
    - reasons: list of strings, key factors influencing assessment

5. Offer constructive feedback:
  - Address significant mismatches

6. Summarize suitability:
  - Concise evaluation of candidate's fit
  - Key factors influencing assessment
  - Decision: "yes" (ideal match), "no" (significant mismatch), or "kiv" (potential despite mismatches)

Guidelines:

- Think step-by-step
- Consider explicit and implicit information
- Maintain objectivity and thoroughness
- Tailor recommendations to optimize chances
- Educations matters more for junior roles, and less important for more senior roles
- technical skills refer to the specific technical term (skills, programming languages, tools, frameworks, etc.) to the job description
- technical skills and experience are the most important factor, followed by education, then experience, and least important is soft skills


Input:

job_description:
{job_description},

resume: 
{resume}

OUTPUT ONLY VALID JSON. NOTHING ELSE.
"""