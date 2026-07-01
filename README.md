# qa-chatbot

## Introduction

qa-chatbot is a simple, lightweight Q&A chatbot built with Streamlit and LangChain that demonstrates how to connect conversational LLMs to a minimal web UI. It provides two example frontends:

- app.py — an OpenAI-backed Streamlit app that uses the ChatOpenAI integration from langchain_openai.
- ollama_app.py — an alternative Streamlit app that uses local/hosted Ollama models via langchain_community.

This project is intended for developers who want a small, easy-to-understand reference for wiring up LLMs to a web interface, experimenting with model selection and basic prompt templates, and testing responses locally.

## Features

- Streamlit-based web UI for quick prototyping and demonstration.
- Two example LLM integrations: OpenAI (ChatOpenAI) and Ollama (Ollama).
- Simple prompt template using LangChain's ChatPromptTemplate.
- Sidebar controls for model selection and response parameters (temperature, max tokens).

## Stack

- Language: Python
- Framework / runtime: Streamlit (for the UI)
- Notable libraries: langchain, langchain_openai, langchain_community, python-dotenv

## Repository layout

```
app.py           # Streamlit app using OpenAI via langchain_openai
ollama_app.py    # Streamlit app using Ollama via langchain_community
requirements.txt # Python dependencies
README.md        # This file
.env             # Example environment variables (not committed by default)
```

## How it works (brief)

Both Streamlit apps define a small ChatPromptTemplate with a system and human message. They construct a LangChain pipeline that composes the prompt, the LLM, and a simple StrOutputParser to produce a string response. The UI collects a question from the user and shows the model's answer in the page.

- app.py: expects an OpenAI API key (entered in the sidebar). It sets some LANGCHAIN/LANGSMITH environment variables and constructs a ChatOpenAI instance.
- ollama_app.py: demonstrates using an Ollama model via langchain_community's Ollama wrapper (useful for local or Ollama-hosted models).

## Quick start

1. Clone the repository:

```bash
git clone https://github.com/ssampathkumar104/qa-chatbot.git
cd qa-chatbot
```

2. Install dependencies (use a virtual environment):

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

3. Create a .env file or set environment variables. Example .env:

```dotenv
# For OpenAI example
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_if_any
LANGCHAIN_PROJECT=MyProject
```

4. Run the OpenAI-backed app:

```bash
streamlit run app.py
```

Or run the Ollama-backed app (if you have Ollama available):

```bash
streamlit run ollama_app.py
```

5. Open the Streamlit URL shown in the terminal (usually http://localhost:8501) and enter your question.

## Environment / notes

- app.py reads OpenAI API credentials from the sidebar input and also references LANGSMITH/LANGCHAIN env vars used by LangChain tracing. Set these if you use LangSmith tracing.
- ollama_app.py assumes you have access to Ollama models (local or hosted) and will attempt to use the model name selected in the sidebar.
- This project is intentionally minimal — treat it as a starting point for experimenting, not a production-ready chatbot.

## Next steps / ideas

- Add conversation history so the assistant can use context across turns.
- Add retrieval-augmented generation (RAG) to answer questions from a knowledge base or documents.
- Add tests and CI for dependency checks and linting.

## License

No license is specified. Add a LICENSE file if you want to open-source this project under a specific license.
