# evaluate resume 
def two_stage_eval_jd(model_tuples: List[Tuple[str, RunnableSequence]], job_description: str, job_id: str, output_dir: str) -> Union[pd.DataFrame, None]:
    model_results = {}
    for model_name, grader in model_tuples:
        try:
            result = grader.invoke({"job_description": job_description})
            model_results[model_name] = result

            # save model result 
            json_file = os.path.join(output_dir, f"{job_id}_{model_name}.json")
            with open(json_file, "w") as f:
                json.dump(result, f, indent=4)
            time.sleep(2.1)  # Add a small delay to avoid rate limiting

        except Exception as e:
            error_msg = f"Error with {model_name} for job_id: {job_id}. Error: {str(e)}"
            logging.error(error_msg)
            print(error_msg)

    if not model_results:
        error_msg = f"All models failed for job_id: {job_id}."
        logging.error(error_msg)
        print(error_msg)
        return None

def two_stage_eval_cv(model_tuples: List[Tuple[str, RunnableSequence]], job_requirements: str, job_id: str, cv: str, cv_id: str, output_dir: str) -> Union[pd.DataFrame, None]:
    model_results = {}
    for model_name, grader in model_tuples:
        try:
            result = grader.invoke({"job_requirements": job_requirements, "resume": cv})
            model_results[model_name] = result

            # save model result 
            json_file = os.path.join(output_dir, f"{job_id}_{cv_id}_{model_name}.json")
            with open(json_file, "w") as f:
                json.dump(result, f, indent=4)
            time.sleep(2.1)  # Add a small delay to avoid rate limiting

        except Exception as e:
            error_msg = f"Error with {model_name} for job_id: {job_id}. Error: {str(e)}"
            logging.error(error_msg)
            print(error_msg)

    if not model_results:
        error_msg = f"All models failed for job_id: {job_id}."
        logging.error(error_msg)
        print(error_msg)
        return None

def evaluate_resumes():
  pass