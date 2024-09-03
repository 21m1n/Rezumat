from typing import Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from ..prompts.two_stage_eval_cv import TWO_STAGE_EVAL_CV_PROMPT
from ..prompts.two_stage_eval_jd import TWO_STAGE_EVAL_JD_PROMPT


def get_model(model_text: str, model_id: str, api_key: str, temperature: float=0, max_tokens: int=2048) -> Tuple[str, RunnableSequence]:
    """get model based on the input data"""

    if model_text == "groq":
      model = ChatGroq(model=model_id, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
    elif model_text == "OpenAI":
      model = ChatOpenAI(model=model_id, temperature=temperature, max_tokens= max_tokens, api_key=api_key)
    elif model_text == "Anthropic":
      model = ChatAnthropic(model=model_id, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
    elif model_text == "Ollama":
      model = ChatOllama(model=model_id, temperature=temperature, max_tokens=max_tokens)
      
    return model
  
def get_eval_chain(model_text: str, model_id: str, api_key: str, eval_type: str):

  model_text = model_text.lower()

  model = get_model(model_text, model_id, api_key)
  
  if eval_type == "jd":
    eval_prompt = PromptTemplate(
      input_variables=["job_description"],
      template=TWO_STAGE_EVAL_JD_PROMPT
      )
  elif eval_type == "cv":
    eval_prompt = PromptTemplate(
      input_variables=["job_requirements", "resume"],
      template=TWO_STAGE_EVAL_CV_PROMPT
      )
  else:
    raise ValueError("Invalid type")
    
  grader = eval_prompt | model | JsonOutputParser()
  
  return (model_text.lower(), grader)
