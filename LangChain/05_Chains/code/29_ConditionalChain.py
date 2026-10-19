from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

parser = StrOutputParser()

class Feedback(BaseModel): # We have created this because without it the llm might return something else so we have enforced the sentiment value to be "positive" or "negative"
    sentiment : Literal['positive', 'negative'] = Field(description = 'Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template = "Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}",
    input_variables = ['feedback'],
    partial_variables= {
        'format_instruction' : parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | llm | parser2

prompt2 = PromptTemplate(
    template = "Write an appropriate response to this positive feedback in short paragraph \n {feedback}",
    input_variables = ['feedback']
)
prompt3 = PromptTemplate(
    template = "Write an appropriate response to this negative feedback in short paragraph \n {feedback}",
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive', prompt2 | llm | parser),
    (lambda x : x.sentiment == 'negative', prompt3 | llm | parser),
    RunnableLambda(lambda x : "Could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke(
    {
        'feedback' : 'This is a terrible phone'
    }
)

chain.get_graph().print_ascii()

print(result)
