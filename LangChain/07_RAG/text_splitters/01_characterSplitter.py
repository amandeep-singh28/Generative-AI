import warnings
warnings.filterwarnings('ignore')

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r'E:\Generative AI\07_RAG\data\sample_3_to_4_page_text.pdf')

text = loader.load()

# text = """
# Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what's possible beyond our planet.

# These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
# """
splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 5,
    separator = ''
)

# result = splitter.split_text(text)
result = splitter.split_documents(text) # If there are 5 pages than 5 documents would be created and than that 5 documents is being passed here

print(result[0].page_content)
print(result[1].page_content)
