import warnings
warnings.filterwarnings('ignore')

from langchain_community.document_loaders import WebBaseLoader

url = "https://docs.langchain.com/oss/python/langchain/tools#tool-execution"

loader = WebBaseLoader(url)
docs = loader.load()

print(docs[0].page_content)