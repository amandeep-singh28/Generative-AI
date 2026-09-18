import warnings
warnings.filterwarnings('ignore')

from langchain_community.document_loaders import CSVLoader

document = CSVLoader(r'E:\Generative AI\07_RAG\data\Position_Salaries.csv')

loader = document.load()

print(loader[0].page_content)
print(loader[1].page_content)
