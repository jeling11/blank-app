import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Load vector database
db = FAISS.load_local("vectordb", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# Define persona-styled system prompt
template = """
You are Paul Golding, a polymath technologist and thought leader known for eloquent writing about AI, systems thinking, hardware-software integration, and innovation strategy. 
Respond to the question using Paul’s tone, voice, and ideas.
Context: {context}
Question: {question}
Answer as Paul:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Load local LLM
llm = Ollama(model="mistral")

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# Streamlit UI
st.set_page_config(page_title="Ask Paul", page_icon="🧠")
st.title("🧠 Ask Paul Golding (RAG Chatbot)")
st.markdown("Ask a question and receive a Paul-style response based on his writings.")

user_input = st.text_area("Your Question", placeholder="e.g., How do you see AI reshaping systems design?")
if st.button("Get Paul's Take") and user_input.strip():
    with st.spinner("Thinking like Paul..."):
        result = qa_chain({"query": user_input})
        st.markdown("### 🧠 Paul's Response")
        st.success(result["result"])
