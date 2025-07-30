import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from openai.error import RateLimitError, AuthenticationError

st.set_page_config(page_title="Paul Golding Persona", page_icon="🧠")

st.title("🧠 Paul Golding Persona")
st.markdown("Simulate responses in the style of Paul Golding using a fine-tuned persona prompt.")

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

model_choice = st.selectbox(
    "Choose Model (requires access for GPT-4)",
    ["gpt-3.5-turbo", "gpt-4"],
    help="Use GPT-4 only if your API key has access. GPT-3.5 is cheaper and more broadly available."
)

system_template = """You are Paul Golding, a polymath technologist and deep thinker.
Your writing style is informed, analytical, philosophical, and eloquently metaphorical. 
You reference technological ecosystems, AI, innovation, systems thinking, hardware/software synthesis, and occasionally historical or biological analogies. 
You prioritize clarity while embracing complexity and nuance."""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{input}")
])

if openai_api_key:
    chat = ChatOpenAI(temperature=0.6, openai_api_key=openai_api_key, model=model_choice)
    chain = LLMChain(llm=chat, prompt=prompt)

    user_input = st.text_area("Enter your question or prompt to Paul", height=200)

    if st.button("Get Response") and user_input:
        with st.spinner("Thinking like Paul..."):
            try:
                response = chain.run({"input": user_input})
                st.markdown("### 🧠 Paul's Response")
                st.write(response)
            except RateLimitError as e:
                st.error("⚠️ Rate limit or quota exceeded. Please check your OpenAI usage and billing settings.")
            except AuthenticationError:
                st.error("🚫 Invalid API key. Please double-check and try again.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
else:
    st.warning("Please enter your OpenAI API key to begin.")
