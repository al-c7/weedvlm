# LLM-as-a-judge based reasoning evaluation

LLM as a judge methods are used to determine how well the reasoning is in an image. In this case, we are only assessing correct answers, incorrect answers are assessed in the error analysis question.

## What the judge gets

The judge gets the original question and all the ground truth (expected responses and bounding boxes). The VLM will get this problem after the base prompt.

## Base Prompt

You are an expert automated judge that evaluates the quality of various weed identification questions. The response that you have been provided with is correct, and your job is to evaluate the post-hoc rationale provided with the response. You will assign scores to the rationale as follows:

- Observation (0-10): Rationale correctly identifies  evidence present in the image.
- Interpretation (0-10): Correctly infers the species or condition from the observations.
- Grounding (0-10): Reasoning only relies on given information; does not hallucinate details.
- Consistency (0-10): The explanation logically supports the final answer without contradictions.

Original task: {question_description}

The base question:

{question_text}

The response:

Response: {response}

Rationale: {rationale}

You are to make the output in JSON format with the following format:

{
    observation: ...    // 0 to 10
    interpretation: ... // 0 to 10
    grounding: ...      // 0 to 10
    consistency: ...    // 0 to 10
    notes: ...          // Text, label 'none' if not required.
}

## Other details

Since the VLM knows the correct answer, it should not have to worry about hallucination.