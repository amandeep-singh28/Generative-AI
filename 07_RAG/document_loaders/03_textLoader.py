import warnings
warnings.filterwarnings('ignore')

from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    path = r'E:\Generative AI\Files',
    glob = '*.txt',
    loader_cls = TextLoader
)

docs = loader.lazy_load()

for documents in docs:
    print(documents.page_content)
    
# print(len(docs))
# print(docs[1])