from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, load_prompt
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

template = load_prompt("template.json")

prompt = template.invoke({
    "paper_input" : paper_input,
    "style_input": style_input,
    "length_input" : length_input
})

user_input = st.text_input("Enter your prompt")

if st.button("Run "):
    result = llm.invoke(prompt)
    st.write(result.content) 