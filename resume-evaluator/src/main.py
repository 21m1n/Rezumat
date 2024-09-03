# 1. read in resume (upload pdf in batches)
# 2. read in job description
# 3. parse pdfs
# 4. read in criteria
# 5. evaluate resume
# 6. return score

from .utils.estimate_cost import hello
from .prompts.two_stage_eval_cv import TWO_STAGE_EVAL_CV_PROMPT
from .prompts.two_stage_eval_jd import TWO_STAGE_EVAL_JD_PROMPT
from .evaluators.resume_evaluator import two_stage_eval_jd, two_stage_eval_cv

hello()

# def main():
#     print("Hello World")


# if __name__ == "__main__":
#     main()