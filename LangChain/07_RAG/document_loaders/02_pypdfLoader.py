import warnings
warnings.filterwarnings('ignore')

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r'E:\Generative AI\07_RAG\data\sample_3_to_4_page_text.pdf')

docs = loader.load()

print(len(docs)) # total number of pages
print(docs[0].page_content)
print(docs[0].metadata)
