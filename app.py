import openai
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
import os

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API client
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an helpful assistant for chatbot."),
        ("human", "{question}"),
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key
    
    llm = ChatOpenAI(model=llm, temperature=temperature, max_tokens=max_tokens)

    output_Parser = StrOutputParser()
    chain = prompt | llm | output_Parser
    answer = chain.invoke({
        "question": question
    })
    return answer

## title of the app
st.title("Q&A Chatbot with OpenAI")

## sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# dropdown to select various openAI models
llm = st.sidebar.selectbox("Select OpenAI Model", ["gpt-4-turbo", "gpt-4o","gpt-4"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=200)

## input box for user question
st.write("Enter your question:")

user_input = st.text_area("You:")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please enter a question to get a response.")
