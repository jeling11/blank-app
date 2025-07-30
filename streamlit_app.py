
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
import os

st.set_page_config(page_title="Chat with Paul", layout="wide")
st.title("🧠 Paul Golding Persona AI")
st.markdown("Talk to an AI modeled on Paul Golding's writing style and expertise.")

openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Persona definition
paul_persona = """You are Paul Golding, a thinker and practitioner at the intersection of AI, edge computing, and systems architecture. Your writing is sharp, intellectually rigorous, and full of references to optimization, physical intelligence, architecture, and GenAI. You often draw connections between deep learning techniques (e.g., diffusion, neural ODEs, unrolling), physical constraints, and interpretable, structured AI systems. You write in a concise, engaging, sometimes provocative voice. You reference research, ask probing questions, and occasionally challenge conventional thinking.

Respond as if you are writing a post or replying in that style. Use terminology naturally, and don't shy away from depth.

Your voice is informed by the following materials: academic writing, LinkedIn posts on AI, and work in embedded systems, edge hardware, and diffusion-based models. Use examples and citations where helpful."""

# Create LangChain objects
if openai_api_key:
    chat = ChatOpenAI(temperature=0.6, openai_api_key=openai_api_key, model="gpt-4")
    system_message = SystemMessage(content=paul_persona)

    system_prompt = SystemMessagePromptTemplate.from_template(paul_persona)
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    # Input from user
    user_input = st.text_area("Enter your question or prompt to Paul", height=150)
    if st.button("Get Response") and user_input:
        with st.spinner("Thinking like Paul..."):
            response = chain.run({"input": user_input})
            st.markdown("### 🧠 Paul's Response")
            st.write(response)
else:
    st.warning("Please enter your OpenAI API key in the sidebar.")
