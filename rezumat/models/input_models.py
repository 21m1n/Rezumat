from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CandidateEvaluationWeights(BaseModel):
    technical_skills: int = Field(ge=0, le=100)
    soft_skills: int = Field(ge=0, le=100)
    experience: int = Field(ge=0, le=100)
    education: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def check_total(self) -> CandidateEvaluationWeights:
        total = sum(self.__dict__.values())
        if total != 100:
            raise ValueError(f"Total weight must be 100%. Current total: {total}%")
        return self


class InputModel(BaseModel):
    text_input: str
    additional_text: str = ""
    input_type: Literal["Text", "File"]
    api_key: str
    interface: Literal["Groq", "OpenAI", "Anthropic"]
    model: Literal["llama3-70b-8192", "gpt-3.5-turbo", "gpt-4"]
    weights: CandidateEvaluationWeights
