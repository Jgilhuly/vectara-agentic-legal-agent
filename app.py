import os
import re
from PIL import Image
import sys
import requests
import json

from omegaconf import OmegaConf
import streamlit as st
from streamlit_pills import pills

from typing import Optional
from pydantic import Field, BaseModel
from vectara_agent.agent import Agent, AgentStatusType
from vectara_agent.tools import ToolsFactory

from dotenv import load_dotenv
load_dotenv(override=True)

initial_prompt = "How can I help you today?"
citation_description = "The citation for a particular case, including the volumn number, reporter and first page. For example: 253 P.2d 136"

def create_tools(cfg):

    def get_opinion_text(
            case_citation = Field(description = citation_description)
            ) -> str:
        """
        Given case citation, returns the full opinion/ruling text of the case.
        If there is more than one opinion for the case, the type of each opinion is returned with the text, and the opinions are separated by semicolons (;)
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

    def get_case_document_pdf(
            case_citation = Field(description = citation_description)
            ) -> str:
        """
        Given a case citation, returns a valid web url to a pdf of the case record
        """

        citation_components = case_citation.split()
        volume_num = citation_components[0]
        reporter = '-'.join(citation_components[1:-1]).replace('.', '').lower()
        first_page = int(citation_components[-1])

        response = requests.get(f"https://static.case.law/{reporter}/{volume_num}/cases/{first_page:04d}-01.json")
        res = json.loads(response.text)
        page_number = res["first_page_order"]

        return f"static.case.law/{reporter}/{volume_num}.pdf#page={page_number}"

    def get_case_document_page(
            case_citation = Field(description = citation_description)
            ) -> str:
        """
        Given a case citation, returns a valid web url to a page with information about the case.
        """

        citation_components = case_citation.split()
        volume_num = citation_components[0]
        reporter = '-'.join(citation_components[1:-1]).replace('.', '').lower()
        first_page = int(citation_components[-1])

        return f"https://case.law/caselaw/?reporter={reporter}&volume={volume_num}&case={first_page:04d}-01"
        
    def get_cited_cases(
            case_citation = Field(description = citation_description)
            ) -> str:
        """
        Given a case citation, returns a list of other cases that are cited by the opinion of this case.
        The output is a list of case citations, separated by semicolons (;)
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

    def validate_url(
            url = Field(description = "A web url pointing to case-law document")
        ) -> bool:
        """
        Given a link, returns whether or not the link is valid.
        If it is not valid, it should not be used in any output.
        """  

        pdf_pattern = re.compile(r'^https://static.case.law/.*')
        document_pattern = re.compile(r'^https://case.law/caselaw/?reporter=.*')
        return bool(pdf_pattern.match(url)) | bool(document_pattern.match(url))

    class QueryCaselawArgs(BaseModel):
        query: str = Field(..., description="The user query.")
        citations: Optional[str] = Field(default = None, 
                                         description = "The citation of the case. Optional.", 
                                         examples = ['253 P.2d 136', '10 Alaska 11', '6 C.M.A. 3'])

    tools_factory = ToolsFactory(vectara_api_key=cfg.api_key, 
                                 vectara_customer_id=cfg.customer_id, 
                                 vectara_corpus_id=cfg.corpus_id)
    ask_caselaw = tools_factory.create_rag_tool(
        tool_name = "ask_caselaw",
        tool_description = """
        Returns a response (str) to the user query base on case law in the state of Alaska.
        If 'citations' is provided, filters the response based on information from that case.
        The response might include metadata about the case such as title/name the ruling, the court, the decision date, and the judges.
        Use this tool for general case law queries.
        """,
        tool_args_schema = QueryCaselawArgs,
        reranker = "multilingual_reranker_v1", rerank_k = 100, 
        n_sentences_before = 2, n_sentences_after = 2, lambda_val = 0.0,
        summary_num_results = 10,
        vectara_summarizer = 'vectara-summary-ext-24-05-med-omni',
        include_citations = False,
    )

    return (tools_factory.get_tools([
            get_opinion_text,
            get_case_document_pdf,
            get_case_document_page,
            get_cited_cases,
            validate_url
        ]) + 
        tools_factory.standard_tools() + 
        tools_factory.legal_tools() + 
        tools_factory.guardrail_tools() +
        [ask_caselaw]
    )

def initialize_agent(_cfg):
    
    legal_assistant_instructions = """
    - You are a helpful legal assistant, with expertise in case law for the state of Alaska.
    - If the user has a legal question that involves long and complex text, 
      break it down into sub-queries and use the ask_caselaw tool to answer each sub-question, 
      then combine the answers to provide a complete response. 
    - If the ask_caselaw tool responds that it does not have enough information to answer the query,
      try to rephrase the query and call the tool again.
    - When presenting the output from ask_caselaw tool,
      Extract metadata from the tool's response, and respond in this format:
      'On <decision date>, the <court> ruled in <case name> that <judges ruling>. This opinion was authored by <judges>'.
    - IMPORTANT: The ask_caselaw tool is your primary tools for finding information about cases. 
      Do not use your own knowledge to answer questions.
    - If two cases have conflicting rulings, assume that the case with the more current ruling date is correct.
    - IMPORTANT: If the response is based on cases that are older than 5 years, make sure to inform the user that the information may be outdated,
      since some case opinions may no longer apply in law.
    - To summarize the text of a case, first use the get_opinion_text to retrieve the full text and then use the summarize_legal_text tool to summarize it.
    - If a user wants to learn more about a case, you can provide them a link to the case record using the get_case_document_pdf tool.
      If this is unsuccessful, you can use the get_case_document_page tool. Don't call the get_case_document_page tool until after you have tried the get_case_document_pdf tool.
      Don't provide links from any other tools!
    - IMPORTANT: The displayed text for this link should be the name_abbreviation of the case (DON'T just say the info can be found here).
    - If a user wants to test their argument, use the ask_caselaw tool to gather information about cases related to their argument 
      and the critique_as_judge tool to determine whether their argument is sound or has issues that must be corrected.
    - When including a link in your response, make sure to validate the link using the validate_url tool.
    - Never discuss politics, and always respond politely.
    """

    def update_func(status_type: AgentStatusType, msg: str):
        if status_type != AgentStatusType.AGENT_UPDATE:
            output = f"{status_type.value} - {msg}"
            st.session_state.log_messages.append(output)

    agent = Agent(
        tools=create_tools(_cfg),
        topic="Case law in Alaska",
        custom_instructions=legal_assistant_instructions,
        update_func=update_func
    )

    return agent


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

def launch_bot():
    def reset():
        st.session_state.messages = [{"role": "assistant", "content": initial_prompt, "avatar": "🦖"}]
        st.session_state.thinking_message = "Agent at work..."
        st.session_state.log_messages = []
        st.session_state.prompt = None
        st.session_state.first_turn = True
        st.session_state.show_logs = False

    st.set_page_config(page_title="Legal Assistant", layout="wide")
    if 'cfg' not in st.session_state:
        cfg = OmegaConf.create({
            'customer_id': str(os.environ['VECTARA_CUSTOMER_ID']),
            'corpus_id': str(os.environ['VECTARA_CORPUS_ID']),
            'api_key': str(os.environ['VECTARA_API_KEY']),
            'examples': os.environ.get('QUERY_EXAMPLES', None)
        })
        st.session_state.cfg = cfg
        st.session_state.ex_prompt = None
        example_messages = [example.strip() for example in cfg.examples.split(",")] if cfg.examples else []
        st.session_state.example_messages = [em for em in example_messages if len(em)>0]
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
                res = res.replace('$', '\\$')  # escape dollar sign for markdown
            message = {"role": "assistant", "content": res, "avatar": '🤖'}
            st.session_state.messages.append(message)
            st.markdown(res)
        st.session_state.ex_prompt = None
        st.session_state.prompt = None
        st.rerun()

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