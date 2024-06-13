
from omegaconf import OmegaConf
import streamlit as st
import os
from PIL import Image
import re
import sys
import datetime

from pydantic import Field, BaseModel
from vectara_agent.agent import Agent, AgentType, AgentStatusType
from vectara_agent.tools import ToolsFactory


tickers = {
    "AAPL": "Apple Computer", 
    "GOOG": "Google", 
    "AMZN": "Amazon",
    "SNOW": "Snowflake",
    "TEAM": "Atlassian",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "MSFT": "Microsoft",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel",
    "NFLX": "Netflix",
}
years = [2020, 2021, 2022, 2023, 2024]
initial_prompt = "How can I help you today?"

def create_tools(cfg):    

    def get_company_info() -> list[str]:
        """
        Returns a dictionary of companies you can query about their financial reports.
        The output is a dictionary of valid ticker symbols mapped to company names.
        You can use this to identify the companies you can query about, and their ticker information.
        """
        return tickers

    def get_valid_years() -> list[str]:
        """
        Returns a list of the years for which financial reports are available.
        """
        return years
    
    class QueryFinancialReportsArgs(BaseModel):
        query: str = Field(..., description="The user query.")
        year: int = Field(..., description=f"The year. an integer between {min(years)} and {max(years)}.")
        ticker: str = Field(..., description=f"The company ticker. Must be a valid ticket symbol from the list {tickers.keys()}.")

    tools_factory = ToolsFactory(vectara_api_key=cfg.api_key, 
                                 vectara_customer_id=cfg.customer_id, 
                                 vectara_corpus_id=cfg.corpus_id)
    query_financial_reports = tools_factory.create_rag_tool(
        tool_name = "query_financial_reports",
        tool_description = """
        Given a company name and year, 
        returns a response (str) to a user query about the company's financial reports for that year.
        make sure to provide the a valid company ticker and year.
        """,
        tool_args_schema = QueryFinancialReportsArgs,
        tool_filter_template = "doc.year = {year} and doc.ticker = '{ticker}'",
        reranker = "multilingual_reranker_v1", rerank_k = 100, 
        n_sentences_before = 2, n_sentences_after = 2, lambda_val = 0.01,
        summary_num_results = 10,
        vectara_summarizer = 'vectara-summary-ext-24-05-med-omni',
    )

    return (tools_factory.get_tools(
                [
                    get_company_info, 
                    get_valid_years,
                ]
            ) +
            tools_factory.standard_tools() + 
            tools_factory.financial_tools() + 
            tools_factory.guardrail_tools() +
            [query_financial_reports]
    )

@st.cache_resource
def initialize_agent(agent_type: AgentType, _cfg):
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    financial_bot_instructions = f"""
    - You are a helpful financial assistant, with expertise in finanal reporting, in conversation with a user. 
    - Today's date is {date}.
    - Report in a concise and clear manner, and provide the most relevant information to the user.
    - Respond in a concise format by using appropriate units of measure (e.g., K for thousands, M for millions, B for billions). 
    - Use tools when available instead of depending on your own knowledge.
    - When querying a tool for a numeric value or KPI, use a concise and non-ambiguous description of what you are looking for. 
    - If you calculate a metric, make sure you have all the necessary information to complete the calculation. Don't guess.
    - Be very careful not to report results you are not confident about.
    """

    def update_func(status_type: AgentStatusType, msg: str):
        output = f"{status_type.value} - {msg}"
        st.session_state.thinking_placeholder.text(output)

    agent = Agent(
        agent_type=agent_type,
        tools=create_tools(_cfg),
        topic="10-K annual financial reports",
        custom_instructions=financial_bot_instructions,
        update_func=update_func
    )
    return agent

def launch_bot(agent_type: AgentType):
    def reset():
        cfg = st.session_state.cfg
        st.session_state.messages = [{"role": "assistant", "content": initial_prompt, "avatar": "🦖"}]
        st.session_state.thinking_message = "Agent at work..."
        st.session_state.agent = initialize_agent(agent_type, cfg)

    st.set_page_config(page_title="Financial Assistant", layout="wide")
    if 'cfg' not in st.session_state:
        cfg = OmegaConf.create({
            'customer_id': str(os.environ['VECTARA_CUSTOMER_ID']),
            'corpus_id': str(os.environ['VECTARA_CORPUS_ID']),
            'api_key': str(os.environ['VECTARA_API_KEY']),
        })
        st.session_state.cfg = cfg
        reset()
    cfg = st.session_state.cfg

    # left side content
    with st.sidebar:
        image = Image.open('Vectara-logo.png')
        st.image(image, width=250)
        st.markdown("## Welcome to the financial assistant demo.\n\n\n")
        companies = ", ".join(tickers.values())
        st.markdown(
            f"This assistant can help you with any questions about the financials of several companies:\n\n **{companies}**.\n"
        )

        st.markdown("\n\n")
        if st.button('Start Over'):
            reset()

        st.markdown("---")
        st.markdown(
            "## How this works?\n"
            "This app was built with [Vectara](https://vectara.com).\n\n"
            "It demonstrates the use of Agentic Chat functionality with Vectara"
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
        with st.chat_message("user", avatar='🧑‍💻'):
            print(f"Starting new question: {prompt}\n")
            st.write(prompt)
    
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar='🤖'):
            with st.spinner(st.session_state.thinking_message):
                st.session_state.thinking_placeholder = st.empty()
                res = st.session_state.agent.chat(prompt)
                cleaned = re.sub(r'\[\d+\]', '', res).replace('$', '\\$')
            message = {"role": "assistant", "content": cleaned, "avatar": '🤖'}
            st.session_state.messages.append(message)
            st.session_state.thinking_placeholder.empty()
            st.rerun()

    sys.stdout.flush()

if __name__ == "__main__":
    launch_bot(agent_type=AgentType.OPENAI)
    