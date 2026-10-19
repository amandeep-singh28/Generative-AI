from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text \n {text}",
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = "Generate 5 short questions from the following text \n {text}",
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \n notes -> {notes} & quiz -> {quiz}",
    input_variables = ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'notes' : prompt1 | llm | parser,
        'quiz' : prompt2 | llm | parser
    }
)

merge_chain = prompt3 | llm | parser

chain = parallel_chain | merge_chain

text = """
sklearn.neighbors provides functionality for unsupervised and supervised neighbors-based learning methods. Unsupervised nearest neighbors is the foundation of many other learning methods, notably manifold learning and spectral clustering. Supervised neighbors-based learning comes in two flavors: classification for data with discrete labels, and regression for data with continuous labels.

The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning). The distance can, in general, be any metric measure: standard Euclidean distance is the most common choice. Neighbors-based methods are known as non-generalizing machine learning methods, since they simply “remember” all of its training data (possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).

Despite its simplicity, nearest neighbors has been successful in a large number of classification and regression problems, including handwritten digits and satellite image scenes. Being a non-parametric method, it is often successful in classification situations where the decision boundary is very irregular.

The classes in sklearn.neighbors can handle either NumPy arrays or scipy.sparse matrices as input. For dense matrices, a large number of possible distance metrics are supported. For sparse matrices, arbitrary Minkowski metrics are supported for searches.

There are many learning routines which rely on nearest neighbors at their core. One example is kernel density estimation, discussed in the density estimation section.
"""

result = chain.invoke(
    {'text' : text}
)

chain.get_graph().print_ascii()

print(result)