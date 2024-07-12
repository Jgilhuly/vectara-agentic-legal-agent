
from omegaconf import OmegaConf
import streamlit as st
import os
from PIL import Image
import sys
import pandas as pd
import requests
from dotenv import load_dotenv

from pydantic import Field, BaseModel
from vectara_agent.agent import Agent, AgentStatusType
from vectara_agent.tools import ToolsFactory


load_dotenv(override=True)
initial_prompt = "How can I help you today?"


def create_tools(cfg):    
    
    class QueryCaselawArgs(BaseModel):
        query: str = Field(..., description="The user query.")

    tools_factory = ToolsFactory(vectara_api_key=cfg.api_key, 
                                 vectara_customer_id=cfg.customer_id, 
                                 vectara_corpus_id=cfg.corpus_id)
    ask_caselaw = tools_factory.create_rag_tool(
        tool_name = "ask_caselaw",
        tool_description = """
        Responds to questions about Case Law in the state of Alaska.
        """,
        tool_args_schema = QueryCaselawArgs,
        reranker = "multilingual_reranker_v1", rerank_k = 100, 
        n_sentences_before = 2, n_sentences_after = 2, lambda_val = 0.005,
        summary_num_results = 10,
        vectara_summarizer = 'vectara-summary-ext-24-05-med-omni',
        include_citations = True,
    )

    return (tools_factory.standard_tools() + 
            tools_factory.legal_tools() + 
            tools_factory.guardrail_tools() +
            [ask_caselaw]
    )

def initialize_agent(_cfg):
    legal_bot_instructions = """
    - You are a helpful legal assistant, with expertise in case law for the state of Alaska.
    - Never discuss politics, and always respond politely.
    """

    def update_func(status_type: AgentStatusType, msg: str):
        output = f"{status_type.value} - {msg}"
        st.session_state.log_messages.append(output)

    agent = Agent(
        tools=create_tools(_cfg),
        topic="case law in Alaska",
        custom_instructions=legal_bot_instructions,
        update_func=update_func
    )
    return agent


def toggle_logs():
    st.session_state.show_logs = not st.session_state.show_logs

def launch_bot():
    def reset():
        st.session_state.messages = [{"role": "assistant", "content": initial_prompt, "avatar": "🦖"}]
        st.session_state.thinking_message = "Agent at work..."
        st.session_state.log_messages = []
        st.session_state.prompt = None
        st.session_state.show_logs = False

    st.set_page_config(page_title="Legal Assistant", layout="wide")
    if 'cfg' not in st.session_state:
        cfg = OmegaConf.create({
            'customer_id': str(os.environ['VECTARA_CUSTOMER_ID']),
            'corpus_id': str(os.environ['VECTARA_CORPUS_ID']),
            'api_key': str(os.environ['VECTARA_API_KEY']),
        })
        st.session_state.cfg = cfg
        reset()

    cfg = st.session_state.cfg
    if 'agent' not in st.session_state:
        st.session_state.agent = initialize_agent(cfg)

    # left side content
    with st.sidebar:
        image = Image.open('Vectara-logo.png')
        st.image(image, width=250)
        st.markdown("## Welcome to the Legal assistant demo.\n\n\n")

        st.markdown("\n\n")
        bc1, _ = st.columns([1, 1])
        with bc1:
            if st.button('Start Over'):
                reset()

        st.markdown("---")
        st.markdown(
            "## How this works?\n"
            "This app was built with [Vectara](https://vectara.com).\n\n"
            "It demonstrates the use of Agentic RAG functionality with Vectara"
        )
        st.markdown("---")

    if "messages" not in st.session_state.keys():
        reset()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": '🧑‍💻'})
        st.session_state.prompt = prompt  # Save the prompt in session state
        st.session_state.log_messages = []
        st.session_state.show_logs = False
        with st.chat_message("user", avatar='🧑‍💻'):
            print(f"Starting new question: {prompt}\n")
            st.write(prompt)
        
    # Generate a new response if last message is not from assistant
    if st.session_state.prompt:
        with st.chat_message("assistant", avatar='🤖'):
            with st.spinner(st.session_state.thinking_message):
                res = st.session_state.agent.chat(st.session_state.prompt)
                res = res.replace('$', '\\$')  # escape dollar sign for markdown
            message = {"role": "assistant", "content": res, "avatar": '🤖'}
            st.session_state.messages.append(message)
            st.markdown(res)
            st.session_state.prompt = None

    log_placeholder = st.empty()
    with log_placeholder.container():
        if st.session_state.show_logs:
            st.button("Hide Logs", on_click=toggle_logs)
            for msg in st.session_state.log_messages:
                st.write(msg)
        else:
            if len(st.session_state.log_messages) > 0:
                st.button("Show Logs", on_click=toggle_logs)

    sys.stdout.flush()

if __name__ == "__main__":
    launch_bot()
    