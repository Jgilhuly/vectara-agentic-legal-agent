from PIL import Image
import sys
import uuid

import nest_asyncio
import asyncio

import streamlit as st
from streamlit_pills import pills
from streamlit_feedback import streamlit_feedback

from vectara_agentic.agent import AgentStatusType

from agent import initialize_agent, get_agent_config
from utils import thumbs_feedback, escape_dollars_outside_latex, send_amplitude_data

initial_prompt = "How can I help you today?"

# Setup for HTTP API Calls to Amplitude Analytics
if 'device_id' not in st.session_state:
    st.session_state.device_id = str(uuid.uuid4())

if "feedback_key" not in st.session_state:
        st.session_state.feedback_key = 0

def toggle_logs():
    st.session_state.show_logs = not st.session_state.show_logs

def show_example_questions():        
    if len(st.session_state.example_messages) > 0 and st.session_state.first_turn:            
        selected_example = pills("Queries to Try:", st.session_state.example_messages, index=None)
        if selected_example:
            st.session_state.ex_prompt = selected_example
            st.session_state.first_turn = False
            return True
    return False

def update_func(status_type: AgentStatusType, msg: str):
    if status_type != AgentStatusType.AGENT_UPDATE:
        output = f"{status_type.value} - {msg}"
        st.session_state.log_messages.append(output)

async def launch_bot():
    def reset():
        st.session_state.messages = [{"role": "assistant", "content": initial_prompt, "avatar": "🦖"}]
        st.session_state.thinking_message = "Agent at work..."
        st.session_state.log_messages = []
        st.session_state.prompt = None
        st.session_state.ex_prompt = None
        st.session_state.first_turn = True
        st.session_state.show_logs = False
        if 'agent' not in st.session_state:
            st.session_state.agent = initialize_agent(cfg, update_func=update_func)

    if 'cfg' not in st.session_state:
        cfg = get_agent_config()
        st.session_state.cfg = cfg
        st.session_state.ex_prompt = None
        example_messages = [example.strip() for example in cfg.examples.split(",")] if cfg.examples else []
        st.session_state.example_messages = [em for em in example_messages if len(em)>0]
        reset()

    cfg = st.session_state.cfg

    # left side content
    with st.sidebar:
        image = Image.open('Vectara-logo.png')
        st.image(image, width=175)
        st.markdown(f"## {cfg['demo_welcome']}")
        st.markdown(f"{cfg['demo_description']}")

        st.markdown("\n\n")
        bc1, _ = st.columns([1, 1])
        with bc1:
            if st.button('Start Over'):
                reset()
                st.rerun()

        st.divider()
        st.markdown(
            "## How this works?\n"
            "This app was built with [Vectara](https://vectara.com).\n\n"
            "It demonstrates the use of Agentic RAG functionality with Vectara"
        )

    if "messages" not in st.session_state.keys():
        reset()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    example_container = st.empty()
    with example_container:
        if show_example_questions():
            example_container.empty()
            st.session_state.first_turn = False
            st.rerun()

    # User-provided prompt
    if st.session_state.ex_prompt:
        prompt = st.session_state.ex_prompt
    else:
        prompt = st.chat_input()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": '🧑‍💻'})
        st.session_state.prompt = prompt  # Save the prompt in session state
        st.session_state.log_messages = []
        st.session_state.show_logs = False
        with st.chat_message("user", avatar='🧑‍💻'):
            print(f"Starting new question: {prompt}\n")
            st.write(prompt)
        st.session_state.ex_prompt = None
        
    # Generate a new response if last message is not from assistant
    if st.session_state.prompt:
        with st.chat_message("assistant", avatar='🤖'):
            with st.spinner(st.session_state.thinking_message):
                res = st.session_state.agent.chat(st.session_state.prompt)
                res = escape_dollars_outside_latex(res)
            message = {"role": "assistant", "content": res, "avatar": '🤖'}
            st.session_state.messages.append(message)
            st.markdown(res)

        send_amplitude_data(
            user_query=st.session_state.messages[-2]["content"], 
            bot_response=st.session_state.messages[-1]["content"],
            demo_name=cfg['demo_name']
        )

        st.session_state.ex_prompt = None
        st.session_state.prompt = None
        st.session_state.first_turn = False
        st.rerun()

    # Record user feedback
    if (st.session_state.messages[-1]["role"] == "assistant") & (st.session_state.messages[-1]["content"] != "How can I help you today?"):
        streamlit_feedback(
            feedback_type="thumbs", on_submit = thumbs_feedback, key = st.session_state.feedback_key,
            kwargs = {"user_query": st.session_state.messages[-2]["content"],
                      "bot_response": st.session_state.messages[-1]["content"],
                      "demo_name": cfg["demo_name"]}
        )

    log_placeholder = st.empty()
    with log_placeholder.container():
        if st.session_state.show_logs:
            st.button("Hide Logs", on_click=toggle_logs)
            for msg in st.session_state.log_messages:
                st.text(msg)
        else:
            if len(st.session_state.log_messages) > 0:
                st.button("Show Logs", on_click=toggle_logs)

    sys.stdout.flush()

if __name__ == "__main__":
    st.set_page_config(page_title="Legal Assistant", layout="wide")
    nest_asyncio.apply()
    asyncio.run(launch_bot())
