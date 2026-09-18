from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field   

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation"
)

model = ChatHuggingFace(
    llm = llm
)

class Person(BaseModel):
    name : str = Field(description = "Name of the person")
    age : int = Field(gt = 18, description = "Age of the person")
    city : str = Field(description = "Name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object = Person)

template = PromptTemplate(
    template = "Generate the name, age and city of a famous person of {place} \n {format_instruction}",
    input_variables = ['place'],
    partial_variables = {'format_instruction' : parser.get_format_instructions} 
)
chain = template | model | parser
result = chain.invoke(
    {"place" : "Delhi"}
)
print(result)