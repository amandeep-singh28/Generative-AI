import warnings, os
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


loader = TextLoader(r'E:\Generative AI\Files\company.txt', encoding = 'utf-8') # for special characters

docs = loader.load()

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = "Write a summary on the following text {document}",
    input_variables = ['document'] 
)

chain = prompt | llm | parser

result = chain.invoke(
    {
        'document' : docs[0].page_content
    }
)

print(result)

# print(docs)
# print(docs[0])
# print(docs[0].metadata)
# print(docs[0].page_content)