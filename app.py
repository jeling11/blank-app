import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# Load embedded data
db = FAISS.load_local("vectordb", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# Define persona prompt
template = """
You are Paul Golding, a polymath technologist and thought leader known for eloquent writing about AI, systems thinking, hardware-software integration, and innovation strategy. 
Respond to the question using Paul’s tone, voice, and ideas.
Context: {context}
Question: {question}
Answer as Paul:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Load local LLM (e.g., mistral via Ollama)
llm = Ollama(model="mistral")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Interactive CLI
while True:
    query = input("\nAsk Paul a question (or 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa_chain({"query": query})
    print("\n🧠 Paul says:\n", result["result"])
