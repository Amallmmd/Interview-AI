import json
# from pprint import pprint
from typing import Literal, Union, List
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from dataclasses import dataclass
from dotenv import load_dotenv
import os
load_dotenv()


openai_key = os.getenv("OPENAI_API_KEY")
@dataclass
class Question:
    question: str
    type: Literal["personal", "behavioral", "situational"]
@dataclass
class Evaluation():
    evaluation: Literal["good", "average", "bad"]
    feedback: str | None
    reason: str | None
    samples: list[str] | None

class QuestionGeneratorAgent:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=openai_key)
        self.system_prompt = """You are a non-technical interviewer that interviews \
across the following categories:
- personal
- behavioral
- situational

You will be provided with a candidate's description.

Generate {n_questions} questions, ensuring that there is a question for each category \
and the questions should be based on the candidate's description.

* You answer strictly as a list of JSON objects. Don't include any other verbose texts, \
and don't include the markdown syntax anywhere.

JSON format:
[
    {{"question": "<personal_question>", "type": "personal"}},
    {{"question": "<behavioral_question>", "type": "behavioral"}},
    {{"question": "<situational_question>", "type": "situational"}},
    ...more questions to make up {n_questions} questions
]"""

        self.user_prompt = "Candidate Description:\n{description}"

    # def __call__(self, description: str, n_questions: int = 3) -> list[Question] | None:
    #     questions: list[Question] | None = self.generate_questions(description, n_questions)

    #     return questions

    def run(self, description: str, n_questions: int = 3) -> list[Question] | None:
        return self.generate_questions(description, n_questions)

        

    def generate_questions(self, description: str, n_questions: int = 3) -> Union[List[Question], None]:
        try:
            # Ensure that there are at least 3 questions
            if n_questions < 3:
                n_questions = 3

            output = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt.format(n_questions=n_questions),
                    },
                    {
                        "role": "user",
                        "content": self.user_prompt.format(description=description),
                    },
                ],
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            if output.choices[0].finish_reason != "stop":
                return None

            if not output.choices[0].message.content:
                return None

            # questions_json = output.choices[0].message.content
            # questions = [Question(**q) for q in questions_json]
            questions: list[Question] = json.loads(output.choices[0].message.content)
            return questions
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


# class QuestionGeneratorAgent:
#     def __init__(self) -> None:
#         self.client = OpenAI(api_key=openai_key)
#         self.system_prompt = """You are a non-technical interviewer that interviews \
# across the following categories:
# - personal

# - behavioural
# - situational

# You will be provided with a candidate's description.

# Generate {n_questions} questions, ensuring that there is a question for each category \
# and the questions should be based on the candidate's description.

# * You answer strictly as a list of JSON objects. Don't include any other verbose texts, \
# and don't include the markdown syntax anywhere.

# JSON format:
# [
#     {{"question": "<personal_question>", "type": "personal"}},
   
#     {{"question": "<behavioural_question>", "type": "behavioural"}},
#     {{"question": "<situational_question>", "type": "situational"}},
#     ...more questions to make up {n_questions} questions
# ]"""

#         self.user_prompt = "Candidate Description:\n{description}"

#     def __call__(self, description: str, n_questions: int = 3) -> list[Question] | None:
#         questions: list[Question] | None = self._generate(description, n_questions)

#         return questions

#     def run(self, description: str, n_questions: int = 3) -> list[Question] | None:
#         questions: list[Question] | None = self._generate(description, n_questions)

#         return questions

#     def _generate(self, description: str, n_questions: int) -> list[Question] | None:
#         try:
#             # Ensure that there are at least 4 questions
#             if n_questions < 3:
#                 n_questions = 3

#             output: ChatCompletion = self.client.chat.completions.create(
#                 model="gpt-3.5-turbo-1106",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": self.system_prompt.format(n_questions=n_questions),
#                     },
#                     {
#                         "role": "user",
#                         "content": self.user_prompt.format(description=description),
#                     },
#                 ],
#                 temperature=0.5,
#                 max_tokens=1024,
#                 top_p=1,
#                 frequency_penalty=0,
#                 presence_penalty=0,
#             )

#             if output.choices[0].finish_reason != "stop":
#                 return None

#             if not output.choices[0].message.content:
#                 return None

#             questions: list[Question] = json.loads(output.choices[0].message.content)

#             return questions
#         except Exception:
#             return None



    #----------------------------------------------------important one-------------------------------------------------------



class ResponseEvaluationAgent:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=openai_key)
        self.system_prompt = """You are an interviewer evaluating a \
candidate's response to an interview question. Your task is to:
- Evaluate the candidate's response on the scale of "good", "average", and "bad".
- Provide a reason for why it's categorized as good, average, or bad.
- Offer constructive feedback or suggestions for improvement.
- Provide 2 samples of good responses.

You will be provided with an interview question and a candidate response.

Evaluate and provide output in the following JSON format:
{{
    "evaluation": "good, average, or bad",
    "reason": "Reason why it's good, average, or bad",
    "feedback": "Feedback or suggestions for improvement",
    "samples": [
        "<Good response 1>", 
        "<Good response 2>"
    ]
}}"""
        self.user_prompt = """QUESTION:
{question}

RESPONSE: 
{response}"""

    def __call__(self, question: str, response: str) -> Evaluation | None:

        evaluation: Evaluation | None = self._generate(question, response)

        return evaluation

    def run(self, question: str, response: str) -> Evaluation | None:

        evaluation: Evaluation | None = self._generate(question, response)

        return evaluation

    def generate(self, question: str, response: str) -> Evaluation | None:
 
        try:
            output: ChatCompletion = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": self.user_prompt.format(
                            question=question, response=response
                        ),
                    },
                ],
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            if not output.choices[0].message.content:
                return None

            questions: Evaluation = json.loads(output.choices[0].message.content)

            return questions
        except Exception:
            return None
# question_generator = QuestionGeneratorAgent()
# questions: list[Question] | None = question_generator.run(
#     "a machine learning engineer at a startup in San Francisco. I have 3 years of "
#     "experience and I'm looking for a new job.",
#     6,
# )


