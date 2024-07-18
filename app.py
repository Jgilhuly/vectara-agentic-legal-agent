from omegaconf import OmegaConf
import streamlit as st
import os
from PIL import Image
import sys
import pandas as pd
import requests
import json

from pydantic import Field, BaseModel
from vectara_agent.agent import Agent, AgentStatusType
from vectara_agent.tools import ToolsFactory

from dotenv import load_dotenv
load_dotenv(override=True)

initial_prompt = "How can I help you today?"


def create_tools(cfg):

    
    def get_opinion_text(case_citation) -> str:
        """
        Given a specific case, this tool returns the full opinion/ruling text of the case.
        The input for this function is the "citations" from the citation_metadata for a previous response from the ask_caselaw tool.
        If there is more than one opinion for the case, the type of each opinion is returned with the text, and the opinions are separated by semicolons (;)
        You can use this tool when a user wants a summary or the full text of an opinion for a case.
        """

        citation_components = case_citation.split()
        volume_num = citation_components[0]
        reporter = '-'.join(citation_components[1:-1]).replace('.', '').lower()
        first_page = int(citation_components[-1])

        response = requests.get(f"https://static.case.law/{reporter}/{volume_num}/cases/{first_page:04d}-01.json")
        res = json.loads(response.text)

        if len(res["casebody"]["opinions"]) == 1:
            return res["casebody"]["opinions"][0]["text"]
        else:
            opinions = ""

            for opinion in res["casebody"]["opinions"]:
                opinions += f"Opinion type: {opinion['type']}, text: {opinion['text']};"

            return opinions

    # THIS MAY NOT COMPILE, PROBABLY NEED TO PUT BELOW TOOLS FACTORY DEFINITON
    def summarize_opinion_text(full_text) -> str:
        """
        Given the full opinion text, this tool returns a summary of the opinion.
        The input for this function is the full opinion text.
        Call this function after the get_opinion_text tool is used when a user wants a summary of the opinion text.
        Use the output of the get_opinion_text tool as the input for this function.
        """

        legal_summarizer = tools_factory.legal_tools()[0]

        return legal_summarizer(full_text)

    def get_case_document_pdf(case_citation) -> str:
        """
        Given a specific case, this tool returns a link/url to a pdf of the case record.
        The input for this function is the "citations" from the citation_metadata for a previous response from the ask_caselaw tool.
        You can use this tool when a user wants to read the original text or see more details about a case.
        """

        citation_components = case_citation.split()
        volume_num = citation_components[0]
        reporter = '-'.join(citation_components[1:-1]).replace('.', '').lower()
        first_page = int(citation_components[-1])

        response = requests.get(f"https://static.case.law/{reporter}/{volume_num}/cases/{first_page:04d}-01.json")
        res = json.loads(response.text)
        page_number = res["first_page_order"]

        return f"static.case.law/{reporter}/{volume_num}.pdf#page={page_number}"

    def get_case_document_page(case_citation) -> str:
        """
        Given a specific case, this tool returns a link/url to a page with information about the case.
        The input for this function is the "citations" from the citation_metadata for a previous response from the ask_caselaw tool.
        You can use this tool when a user wants to read the original text or see more details about a case and accessing the pdf is unsuccessful.
        """

        citation_components = case_citation.split()
        volume_num = citation_components[0]
        reporter = '-'.join(citation_components[1:-1]).replace('.', '').lower()
        first_page = int(citation_components[-1])

        # response = requests.get(f"https://static.case.law/{reporter}/{volume_num}/cases/{first_page:04d}-01.json")
        # res = json.loads(response.text)
        # page_number = res["first_page_order"]

        return f"https://case.law/caselaw/?reporter={reporter}&volume={volume_num}&case={first_page:04d}-01"
        
    def get_cited_cases(case_citation) -> str:
        """
        Given a specific case, this tool returns a list of other cases that are cited by the opinion of this case.
        The input for this function is the "citations" from the citation_metadata for a previous response from the ask_caselaw tool.
        The output is a list of case citations, separated by semicolons (;)
        You can use this tool when a user wants to know what other cases were used to form the opinion for a particular case.
        """

        citation_components = case_citation.split()
        volume_num = citation_components[0]
        reporter = '-'.join(citation_components[1:-1]).replace('.', '').lower()
        first_page = int(citation_components[-1])

        response = requests.get(f"https://static.case.law/{reporter}/{volume_num}/cases/{first_page:04d}-01.json")
        res = json.loads(response.text)

        citations = res["cites_to"]

        other_citations = ""

        for citation in citations[:5]:
            other_citations += citation["cite"] + ';'

        return other_citations

    class QueryCaselawArgs(BaseModel):
        query: str = Field(..., description="The user query.")

    tools_factory = ToolsFactory(vectara_api_key=cfg.api_key, 
                                 vectara_customer_id=cfg.customer_id, 
                                 vectara_corpus_id=cfg.corpus_id)
    ask_caselaw = tools_factory.create_rag_tool(
        tool_name = "ask_caselaw",
        tool_description = """
        Returns a response (str) to a user question about case law in the state of Alaska.
        The response might include metadata about the case such as title/name the ruling, the court, the decision date, and the judges.
        You can ask this tool any question regarding case law, and it is specifically designed to answer questions based on semantic meaning of the query.
        The tool will say "I do not have enough information to answer the question accurately"
        """,
        tool_args_schema = QueryCaselawArgs,
        reranker = "multilingual_reranker_v1", rerank_k = 100, 
        n_sentences_before = 2, n_sentences_after = 2, lambda_val = 0.0,
        summary_num_results = 10,
        vectara_summarizer = 'vectara-summary-ext-24-05-med-omni',
        include_citations = True,
    )
    ask_caselaw_keyword = tools_factory.create_rag_tool(
        tool_name = "ask_caselaw_keyword",
        tool_description = """
        Returns a response (str) to a user question about case law in the state of Alaska.
        The response might include metadata about the case such as title/name the ruling, the court, the decision date, and the judges.
        You can ask this tool any question regarding case law, and it is specifically designed to pick up specific keyword in the query for its response.
        The tool will say "I do not have enough information to answer the question accurately"
        """,
        tool_args_schema = QueryCaselawArgs,
        reranker = "multilingual_reranker_v1", rerank_k = 100, 
        n_sentences_before = 2, n_sentences_after = 2, lambda_val = 0.05,
        summary_num_results = 10,
        vectara_summarizer = 'vectara-summary-ext-24-05-med-omni',
        include_citations = True,
    )

    return (tools_factory.get_tools([
        get_opinion_text,
        summarize_opinion_text,
        get_case_document_pdf,
        get_case_document_page,
        get_cited_cases
        ]
        ) + 
            tools_factory.standard_tools() + 
            tools_factory.legal_tools() + 
            tools_factory.guardrail_tools() +
            [ask_caselaw, ask_caselaw_keyword]
    )

def initialize_agent(_cfg):
    legal_bot_instructions = """
    - You are a helpful legal assistant, with expertise in case law for the state of Alaska.
    - Always try to find the most recent cases so that you can provide information regarding the most up-to-date laws. 
    - If the user has a legal question that involves long and complex text, 
      break it down into sub-queries and use the ask_caselaw or ask_caselaw_keyword tools to answer each sub-question, 
      then combine the answers to provide a complete response.
    - IMPORTANT: If the ask_caselaw or ask_caselaw_keyword tools respond that they do not have enough information to answer the query,
      try to use another tool or rephrase the query.
    - IMPORTANT: The ask_caselaw and ask_caselaw_keyword tools are your primary tools for finding information about cases. Do not use your own knowledge to answer questions.
    - If two cases have conflicting rulings, assume that the case with the more current ruling date is correct.
    - When presenting the output from ask_caselaw or ask_caselaw_keyword tools to the user, this is a good format to use where aprpropriate:
      'On {decision date}, the {court} ruled in {case name} that {judges ruling}. This opinion was authored by {judges}'.
    - If a user wants to learn more about a case, you can provide them a link to case record using the get_case_document_pdf tool. 
      If this is unsuccessful, you can use the get_case_document_page tool.
      IMPORTANT: The displayed text for this link should be the name_abbreviation of the case (DON'T just say the info can be found here). 
      Only provide this when prompted to do so by the user.
    - If a user wants a summary of a case opinion, use the get_opinion_text tool to get the full opinion text.
      Then take the output from get_opinion_text tool to call the summarize_opinion_text tool to summarize this opinion. Return this summary to the user.
      If there is more than one opinion for the case, then call the summarize_opinion_text tool for each opinion. 
      (Each opinion will be separated by a semicolon and will begin with the opinion type followed by the text of the opinion)
    - If a user wants to know other cases that were used to draft an opinion, use the get_cited_cases tool to acquire the case citations for those cases.
      With each of these case citations, use the output of the get_cited_cases tool as the input to the ask_caselaw or ask_caselaw_keyword tools to get information about the cases. 
      IMPORTANT: If the query to the ask_caselaw tool says that it does not have enough information, it means that the case is not in our data. To get information about these cases, use the summarize_opinion_text tool and get_case_document tools to get information about the case.
      Make sure to do this before simply returning the results of the query to the user.
    - If a user wants to test their argument, use the ask_caselaw or ask_caselaw_keyword tools to gather information about cases related to their argument 
      and the critique_as_judge tool to determine whether their argument is sound or has issues that must be corrected.
    - Never discuss politics, and always respond politely.
    """

    def update_func(status_type: AgentStatusType, msg: str):
        if status_type != AgentStatusType.AGENT_UPDATE:
            output = f"{status_type.value} - {msg}"
            st.session_state.log_messages.append(output)

    agent = Agent(
        tools=create_tools(_cfg),
        topic="Case law in Alaska",
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
        st.image(image, width=175)
        st.markdown("## Welcome to the Legal assistant demo.\n\n\n")
        st.markdown("This demo can help you prepare for a court case by providing you information about past court cases in Alaska.")

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
                st.text(msg)
        else:
            if len(st.session_state.log_messages) > 0:
                st.button("Show Logs", on_click=toggle_logs)

    sys.stdout.flush()

if __name__ == "__main__":
    launch_bot()