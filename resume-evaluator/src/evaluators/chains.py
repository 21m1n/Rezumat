from typing import Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from ..config import config
from ..prompts.two_stage_eval_cv import TWO_STAGE_EVAL_CV_PROMPT
from ..prompts.two_stage_eval_jd import TWO_STAGE_EVAL_JD_PROMPT

GROQ_API_KEY = config.GROQ_API_KEY
OPENAI_API_KEY = config.OPENAI_API_KEY
ANTHROPIC_API_KEY = config.ANTHROPIC_API_KEY  
TEMPERATURE = config.TEMPERATURE
MAX_TOKENS = config.MAX_TOKENS  

def get_model(model_text: str, model_id: str, temperature: float=0, max_tokens: int=2048, api_key: str=None) -> Tuple[str, RunnableSequence]:
    """get model based on the input data"""

    model_classes = {
      "groq": ChatGroq,
      "openai": ChatOpenAI,
      "anthropic": ChatAnthropic,
      "ollama": ChatOllama
    }
    
    model_text = model_text.lower()
    model_class = model_classes.get(model_text)
    
    if model_class is None:
      raise ValueError(f"Invalid model text: {model_text}")
    
    model = model_class(model=model_id, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
    
    return model
    
def get_eval_chain(model_text: str, model_id: str, api_key: str, eval_type: str):

  model_text = model_text.lower()

  model = get_model(model_text=model_text, model_id=model_id, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, api_key=api_key)

  eval_prompts = {
    "jd": PromptTemplate(
      input_variables=["job_description"],
      template=TWO_STAGE_EVAL_JD_PROMPT
    ),
    "cv": PromptTemplate(
      input_variables=["job_requirements", "resume"],
      template=TWO_STAGE_EVAL_CV_PROMPT
    )
  }
  
  eval_prompt = eval_prompts.get(eval_type)
  
  if eval_prompt is None:
    raise ValueError("Invalid type")
  
  grader = eval_prompt | model | JsonOutputParser()
  
  return (model_text, grader)
