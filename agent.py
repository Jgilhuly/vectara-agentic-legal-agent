import os
import re
import requests
import json
from typing import Tuple, List

from omegaconf import OmegaConf

from typing import Optional
from pydantic import Field, BaseModel

from vectara_agentic.agent import Agent
from vectara_agentic.tools import ToolsFactory, VectaraToolFactory
from vectara_agentic.tools_catalog import summarize_text

from dotenv import load_dotenv
load_dotenv(override=True)

citation_description = '''
    The citation for a particular case. 
    Citation must include the volume number, reporter, and first page. For example: 253 P.2d 136.
'''

def extract_components_from_citation(citation: str) -> Tuple[int, str, int]:
    citation_components = citation.split(' ')
    volume_num = citation_components[0]
    reporter = '-'.join(citation_components[1:-1]).replace('.', '').lower()
    first_page = citation_components[-1]

    if not volume_num.isdigit():
        raise ValueError("volume number must be a number.")
    if not first_page.isdigit():
        raise ValueError("first page number must be a number.")

    return int(volume_num), reporter, int(first_page)

def create_assistant_tools(cfg):

    def get_opinion_text(
            case_citation = Field(description = citation_description),
            summarize: Optional[bool] = False
            ) -> str:
        """
        Given case citation, returns the full opinion/ruling text of the case.
        if summarize is True, the text is summarized.
        If there is more than one opinion for the case, the type of each opinion is returned with the text, 
        and the opinions (or their summaries) are separated by semicolons (;)
        """
        volume_num, reporter, first_page = extract_components_from_citation(case_citation)
        response = requests.get(f"https://static.case.law/{reporter}/{volume_num}/cases/{first_page:04d}-01.json")
        if response.status_code != 200:
            return "Case not found; please check the citation."
        res = json.loads(response.text)

        if len(res["casebody"]["opinions"]) == 1:
            text = res["casebody"]["opinions"][0]["text"]
            output = text if not summarize else summarize_text(text, "law")
        else:
            output = ""
            for opinion in res["casebody"]["opinions"]:
                text = opinion["text"] if not summarize else summarize_text(opinion["text"], "law")
                output += f"Opinion type: {opinion['type']}, text: {text};"
        
        return output

    def get_case_document_pdf(
            case_citation = Field(description = citation_description)
            ) -> str:
        """
        Given a case citation, returns a valid web url to a pdf of the case record
        """
        volume_num, reporter, first_page = extract_components_from_citation(case_citation)
        response = requests.get(f"https://static.case.law/{reporter}/{volume_num}/cases/{first_page:04d}-01.json")
        if response.status_code != 200:
            return "Case not found; please check the citation."
        res = json.loads(response.text)
        page_number = res["first_page_order"]
        return f"https://static.case.law/{reporter}/{volume_num}.pdf#page={page_number}"

    def get_case_document_page(
            case_citation = Field(description = citation_description)
            ) -> str:
        """
        Given a case citation, returns a valid web url to a page with information about the case.
        """
        volume_num, reporter, first_page = extract_components_from_citation(case_citation)
        url = f"https://case.law/caselaw/?reporter={reporter}&volume={volume_num}&case={first_page:04d}-01"
        response = requests.get(url)
        if response.status_code != 200:
            return "Case not found; please check the citation."
        return url
        
    def get_case_name(
            case_citation = Field(description = citation_description)
            ) -> Tuple[str, str]:
        """
        Given a case citation, returns its name and name abbreviation.
        """
        volume_num, reporter, first_page = extract_components_from_citation(case_citation)
        response = requests.get(f"https://static.case.law/{reporter}/{volume_num}/cases/{first_page:04d}-01.json")
        if response.status_code != 200:
            return "Case not found", "Case not found"
        res = json.loads(response.text)
        return res["name"], res["name_abbreviation"]

    def get_cited_cases(
            case_citation = Field(description = citation_description)
            ) -> List[dict]:
        """
        Given a case citation, returns a list of cases that are cited by the opinion of this case.
        The output is a list of cases, each a dict with the citation, name and name_abbreviation of the case.
        """
        volume_num, reporter, first_page = extract_components_from_citation(case_citation)
        response = requests.get(f"https://static.case.law/{reporter}/{volume_num}/cases/{first_page:04d}-01.json")
        if response.status_code != 200:
            return "Case not found; please check the citation."
        res = json.loads(response.text)
        citations = res["cites_to"]
        res = []
        for citation in citations[:10]:
            name, name_abbreviation = get_case_name(citation["cite"])
            res.append({
                "citation": citation["cite"],
                "name": name,
                "name_abbreviation": name_abbreviation
            })
        return res

    def validate_url(
            url = Field(description = "A web url pointing to case-law document")
        ) -> str:
        """
        Given a link, returns whether or not the link is valid.
        If it is not valid, it should not be used in any output.
        """  
        pdf_pattern = re.compile(r'^https://static.case.law/.*')
        document_pattern = re.compile(r'^https://case.law/caselaw/?reporter=.*')
        return "URL is valid" if bool(pdf_pattern.match(url)) | bool(document_pattern.match(url)) else "URL is bad"

    class QueryCaselawArgs(BaseModel):
        query: str = Field(..., description="The user query.")
        citations: Optional[str] = Field(default = None, 
                                         description = "The citations of the case. Optional.",
                                         examples = ['253 P.2d 136', '10 Alaska 11', '6 C.M.A. 3'])

    vec_factory = VectaraToolFactory(vectara_api_key=cfg.api_key, 
                                     vectara_customer_id=cfg.customer_id, 
                                     vectara_corpus_id=cfg.corpus_id)
    tools_factory = ToolsFactory()

    ask_caselaw = vec_factory.create_rag_tool(
        tool_name = "ask_caselaw",
        tool_description = """
        Use this tool for general case law queries.
        Returns a response (str) to the user query base on case law in the state of Alaska.
        If 'citations' is provided, filters the response based on information from that case.
        """,
        tool_args_schema = QueryCaselawArgs,
        reranker = "multilingual_reranker_v1", rerank_k = 100, 
        n_sentences_before = 2, n_sentences_after = 2, lambda_val = 0.0,
        summary_num_results = 10,
        vectara_summarizer = 'vectara-summary-ext-24-05-med-omni',
        include_citations = False,
    )

    return (
        [tools_factory.create_tool(tool) for tool in [
            get_opinion_text,
            get_case_document_pdf,
            get_case_document_page,
            get_cited_cases,
            get_case_name,
            validate_url
        ]] + 
        tools_factory.standard_tools() + 
        tools_factory.legal_tools() + 
        tools_factory.guardrail_tools() +
        [ask_caselaw]
    )

def get_agent_config() -> OmegaConf:
    cfg = OmegaConf.create({
        'customer_id': str(os.environ['VECTARA_CUSTOMER_ID']),
        'corpus_id': str(os.environ['VECTARA_CORPUS_ID']),
        'api_key': str(os.environ['VECTARA_API_KEY']),
        'examples': os.environ.get('QUERY_EXAMPLES', None),
        'demo_name': "legal-agent",
        'demo_welcome': "Welcome to the Legal Assistant demo.",
        'demo_description': "This demo can help you prepare for a court case by providing you information about past court cases in Alaska.",
    })
    return cfg

def initialize_agent(_cfg, update_func=None):
    
    legal_assistant_instructions = """
    - You are a helpful legal assistant, with expertise in case law for the state of Alaska.
    - The ask_caselaw tool is your primary tools for finding information about cases. 
      Do not use your own knowledge to answer questions.
    - For a query with multiple sub-questions, break down the query into the sub-questions, 
      and make separate calls to the ask_caselaw tool to answer each sub-question,
      then combine the answers to provide a complete response.
    - If the ask_caselaw tool responds that it does not have enough information to answer the query,
      try to rephrase the query and call the tool again.
    - When presenting the output from ask_caselaw tool,
      Extract metadata from the tool's response, and respond in this format:
      'On <decision date>, the <court> ruled in <case name> that <judges ruling>. This opinion was authored by <judges>'.
    - Citations include 3 components: volume number, reporter, and first page. 
      Here are some examples: '253 P.2d 136', '10 Alaska 11', '6 C.M.A. 3'
      Never use your internal knowledge to contruct or guess what the citation is.
    - If two cases have conflicting rulings, assume that the case with the more current ruling date is correct.
    - If the response is based on cases that are older than 5 years, make sure to inform the user that the information may be outdated,
      since some case opinions may no longer apply in law.
    - To summarize the case, use the get_opinion_text with summarize set to True.
    - If a user wants to learn more about a case, you can call the get_case_document_pdf tool with the citation to get a valid URL.
      If this is unsuccessful, call the get_case_document_page tool instead. 
      The text displayed with this URL should be the name_abbreviation of the case (DON'T just say the info can be found here).
      Don't call the get_case_document_page tool until after you have tried the get_case_document_pdf tool.
      Don't provide URLs from any other tools. Do not generate URLs yourself.
    - When presenting a URL in your response, use the validate_url tool.
    - If a user wants to test their argument, use the ask_caselaw tool to gather information about cases related to their argument 
      and the critique_as_judge tool to determine whether their argument is sound or has issues that must be corrected.
    - Never discuss politics, and always respond politely.
    """

    agent = Agent(
        tools=create_assistant_tools(_cfg),
        topic="Case law in Alaska",
        custom_instructions=legal_assistant_instructions,
        update_func=update_func
    )

    return agent
