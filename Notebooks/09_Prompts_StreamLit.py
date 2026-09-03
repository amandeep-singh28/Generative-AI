from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import streamlit as st

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

st.header("Researcher Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention is all you need",
        "BERT: Pre training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are few-shot learners",
        "Diffusion models beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
     "Select Explanation Style",
     [
         "Beginner-Friendly",
         "Technical",
         "Code-Oriented",
         "Mathematical"
     ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1 - 2 paragraphs)",
        "Medium (3 - 5 paragraphs)",
        "Long (Detailed explanation)"
    ]
)

template = PromptTemplate(
    template = """
    Please summarize the research paper titled "{paper_input}" with the following specifications:
        Explanation Style: {style_input}
        Explanation Length: {length_input}
        1. Mathematical Details:
            - Include relevant mathematical equations if present in the paper.
            - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
        2. Analogies:
            - Use relatable analogies to simplify the complex ideas.
        If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
        Ensure the summary is clear, accurate and aligned with the provided style and length
""",
    input_variables = ["paper_input", "style_input", "length_input"]
)

prompt = template.invoke({
    "paper_input" : paper_input,
    "style_input": style_input,
    "length_input" : length_input
})

user_input = st.text_input("Enter your prompt")

if st.button("Run "):
    result = llm.invoke(prompt)
    st.write(result.content) 