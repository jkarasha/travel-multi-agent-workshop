"""
LLM-as-Judge Evaluators for Travel Assistant

These evaluators use Azure OpenAI to assess response quality across multiple dimensions.
"""

import sys
from pathlib import Path
from typing import TypedDict

# Add python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from src.app.services.azure_open_ai import get_model


# TypedDict schemas for structured LLM output
class Grade(TypedDict):
    """Boolean grade with reasoning."""
    reasoning: str
    is_correct: bool


class Score(TypedDict):
    """Numeric score (1-5) with reasoning."""
    reasoning: str
    score: int


# LLM-as-judge for answer quality
grader_instructions = """You are evaluating a travel assistant's response quality.

Compare the STUDENT RESPONSE to the REFERENCE RESPONSE for the given QUESTION.

Criteria:
1. Relevance - Does it address the user's travel need?
2. Helpfulness - Does it provide actionable information or ask appropriate follow-up questions?
3. Accuracy - Is the information factually correct (no hallucinations)?
4. Completeness - Does it cover key aspects from the reference?

IMPORTANT:
- The student response can have MORE detail than the reference, as long as it's accurate.
- Follow-up questions for clarification are ENCOURAGED and should be marked as correct.
- Acknowledging preferences and asking for location/details is appropriate behavior.
- The student response should contain actual hotel/restaurant/activity names when recommendations are requested.
- If the student asks clarifying questions instead of making assumptions, this is CORRECT behavior.

Grade as correct (True) if the response meets the criteria and demonstrates good conversational flow, otherwise False."""

# Use the centralized model with structured output
grader_llm = get_model().with_structured_output(Grade, method="json_schema")


async def answer_quality(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """
    LLM-as-judge evaluator for overall answer quality.
    
    Args:
        inputs: The input question
        outputs: The actual system response
        reference_outputs: The expected reference response
        
    Returns:
        Boolean indicating if the response meets quality criteria
    """
    user_prompt = f"""QUESTION: {inputs['question']}
REFERENCE RESPONSE: {reference_outputs['answer']}
STUDENT RESPONSE: {outputs['answer']}"""
    
    grade = await grader_llm.ainvoke([
        {"role": "system", "content": grader_instructions},
        {"role": "user", "content": user_prompt}
    ])
    return grade["is_correct"]


# LLM-as-judge for correctness
correctness_instructions = """You are evaluating a travel assistant's response correctness.

Compare the STUDENT RESPONSE to the REFERENCE RESPONSE for the given QUESTION.

Criteria:
1. Factual Accuracy - Are all facts, places, and details correct?
2. No Hallucinations - Does the response avoid making up information?
3. Consistency - Is the information internally consistent?
4. Appropriate Behavior - Does it acknowledge user input and ask clarifying questions when needed?

IMPORTANT:
- Asking for clarification (e.g., location, dates) is CORRECT behavior when information is missing.
- Acknowledging user preferences before asking follow-up questions is CORRECT.
- The response doesn't need to match the reference exactly if it's factually accurate.

Grade as correct (True) if the response is factually accurate and demonstrates appropriate conversational behavior, otherwise False."""

# Use the centralized model with structured output
correctness_llm = get_model().with_structured_output(Grade, method="json_schema")


async def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """
    LLM-as-judge evaluator for factual correctness only.
    
    Args:
        inputs: The input question
        outputs: The actual system response
        reference_outputs: The expected reference response
        
    Returns:
        Boolean indicating if the response is factually correct
    """
    user_prompt = f"""QUESTION: {inputs['question']}
REFERENCE RESPONSE: {reference_outputs['answer']}
STUDENT RESPONSE: {outputs['answer']}"""
    
    grade = await correctness_llm.ainvoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": user_prompt}
    ])
    return grade["is_correct"]


# LLM-as-judge for humanness (1-5 scale)
humanness_instructions = """You are evaluating a travel assistant's response for humanness and natural conversation quality.

Evaluate the STUDENT RESPONSE for the given QUESTION on a scale of 1-5.

Criteria:
1. Natural Language - Does it sound like a human assistant, not robotic?
2. Appropriate Tone - Is the tone friendly, helpful, and conversational?
3. Empathy - Does it acknowledge user needs and preferences appropriately?
4. Clarity - Is the response clear and easy to understand?

Scoring:
- 5: Exceptionally natural, warm, and human-like
- 4: Very natural with good conversational tone
- 3: Acceptable, somewhat conversational but could be warmer
- 2: Somewhat robotic or impersonal
- 1: Very robotic, cold, or unhelpful tone"""

# Use the centralized model with structured output for scoring
humanness_llm = get_model().with_structured_output(Score, method="json_schema")


async def humanness(inputs: dict, outputs: dict, reference_outputs: dict) -> int:
    """
    LLM-as-judge evaluator for humanness and conversational quality.
    
    Args:
        inputs: The input question
        outputs: The actual system response
        reference_outputs: Not used for humanness evaluation
        
    Returns:
        Integer score from 1-5 indicating conversational quality
    """
    user_prompt = f"""QUESTION: {inputs['question']}
STUDENT RESPONSE: {outputs['answer']}"""
    
    score_result = await humanness_llm.ainvoke([
        {"role": "system", "content": humanness_instructions},
        {"role": "user", "content": user_prompt}
    ])
    return score_result["score"]
