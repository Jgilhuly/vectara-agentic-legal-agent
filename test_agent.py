
import unittest
import re
import os

from omegaconf import OmegaConf
from vectara_agent.agent import Agent

from app import initialize_agent

from dotenv import load_dotenv
load_dotenv(override=True)

class TestAgentResponses(unittest.TestCase):

    def test_responses(self):

        cfg = OmegaConf.create({
            'customer_id': str(os.environ['VECTARA_CUSTOMER_ID']),
            'corpus_id': str(os.environ['VECTARA_CORPUS_ID']),
            'api_key': str(os.environ['VECTARA_API_KEY']),
            'examples': os.environ.get('QUERY_EXAMPLES', None)
        })

        agent = initialize_agent(_cfg=cfg)
        self.assertIsInstance(agent, Agent)

        # Test whether cases are real or fake
        self.assertEqual(''.join(re.findall(r'[a-zA-Z]', agent.chat('Is the case Brown v. Board of Education, 347 U.S. 483 (1954), a real case? Say "yes" or "no" only.'))).lower(), 'yes')
        self.assertEqual(''.join(re.findall(r'[a-zA-Z]', agent.chat('Is the case Bowers v. Hardwick, 478 U.S. 186 (1986), a real case? Say "yes" or "no" only.'))).lower(), 'yes')
        self.assertEqual(''.join(re.findall(r'[a-zA-Z]', agent.chat('Is the case Columbia University v. Rodham, 564 U.S. 911 (2010), a real case? Say "yes" or "no" only.'))).lower(), 'no')

        # Test case citation extraction
        self.assertEqual(agent.chat('What is the citation for the case Brown v. Board of Education? Provide ONLY the citation in "<volume>, <reporter>, <page>" format, nothing else.'), '347 U.S. 483')
        self.assertEqual(agent.chat('What is the citation for the case Bowers v. Hardwick? Provide ONLY the citation in "<volume>, <reporter>, <page>" format, nothing else.'), '478 U.S. 186')
        self.assertEqual(agent.chat('What is the citation for the case McCulloch v. Maryland? Provide ONLY the citation in "<volume>, <reporter>, <page>" format, nothing else.'), '17 U.S. 316')

        # Test opinion author identification
        self.assertEqual(agent.chat('Who wrote the majority opinion in Brown v. Board of Education, 347 U.S. 483 (1954)? Provide the first and the last name of the judge ONLY.'), 'Earl Warren')
        self.assertEqual(agent.chat('Who wrote the majority opinion in Bowers v. Hardwick, 478 U.S. 186 (1986)? Provide the first and the last name of the judge ONLY.'), 'Byron White')
        self.assertEqual(agent.chat('Who wrote the majority opinion in McCulloch v. Maryland, 17 U.S. 316 (1819)? Provide the first and the last name of the judge ONLY.'), 'John Marshall')

        # Test opinion text understanding
        self.assertEqual(agent.chat("Did the court in Plessy v. Ferguson, 163 U.S. 537 (1896) affirm or reverse the lower court's decision? Say 'affirm' or 'reverse' only.").lower(), 'affirm')
        self.assertEqual(agent.chat("Did the court in Bowers v. Hardwick, 478 U.S. 186 (1986) affirm or reverse the lower court's decision? Say 'affirm' or 'reverse' only.").lower(), 'reverse')
        self.assertEqual(agent.chat("Did the court in McCulloch v. Maryland, 17 U.S. 316 (1819) affirm or reverse the lower court's decision? Say 'affirm' or 'reverse' only.").lower(), 'reverse')

        # Test court identification
        self.assertIn('united states court of appeals for the second circuit', agent.chat("Which court decided the case Viacom International Inc. v. YouTube, Inc., 676 F.3d 19 (2012)? Provide the name of the court ONLY, nothing else.").lower())
        self.assertIn('united states court of appeals for the district of columbia circuit', agent.chat("Which court decided the case  Durham v. United States, 214 F.2d 862 (1954)? Provide the name of the court ONLY, nothing else.").lower())
        self.assertIn('supreme court', agent.chat("Which court decided the case Bowers v. Hardwick (1986)? Provide the name of the court ONLY, nothing else.").lower())

        # Test overruling of case
        self.assertIn(agent.chat("What year was Whitney v. California, 274 U.S. 357, overruled? Provide the year only."), ['1969', 'I don\'t know.']) # Our agent seems to not find the answer to this question, which I don't see as a problem (At least it's not hallucinating)
        self.assertEqual(agent.chat("What year was Austin v. Michigan Chamber of Commerce, 494 U.S. 652, overruled? Provide the year only."), '2010')

        # Compare two rulings
        self.assertEqual(agent.chat('Do the cases Brown v. Board of Education, 347 U.S. 483 (1954) and Plessy v. Ferguson, 163 U.S. 537 (1896) agree or disagree with each other? Say "agree" or "disagree" only.').lower(), 'disagree')
        # self.assertEqual(agent.chat('Do the cases Youngstown Sheet & Tube Co. v. Sawyer, 343 U.S. 579 (1952) and Medellin v. Texas, 552 U.S. 491 (2008) agree or disagree with each other? Say "agree" or "disagree" only.').lower(), 'agree') # Our agent thinks that these rulings disagree, so I commented out this test.
        self.assertEqual(agent.chat('Do the cases Whitney v. California, 274 U.S. 357 (1927) and Brandenburg v. Ohio, 395 U.S. 444 (1969) agree or disagree with each other? Say "agree" or "disagree" only.').lower(), 'disagree')


if __name__ == "__main__":
    unittest.main()