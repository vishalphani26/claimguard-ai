import unittest
from langchain_core.documents import Document
from pydantic import ValidationError
from insurance_agent import ClaimInput, Decision, validate_citation_ids, _escalation_priority
class AgentTests(unittest.TestCase):
    def test_invalid_claim(self):
        with self.assertRaises(ValidationError): ClaimInput(claim_id='x')
    def test_citations(self):
        d=Decision(verdict='approve',confidence=.9,coverage_summary='Demo',reasoning='Demo',cited_clause_ids=['A','B'],recommended_action='Review')
        self.assertEqual(validate_citation_ids(d,[Document(page_content='Demo',metadata={'clause_id':'A'})]),['B'])
    def test_priority(self):
        self.assertEqual(_escalation_priority({'claim_amount':25000}),'high')
        self.assertEqual(_escalation_priority({'claim_amount':100}),'normal')
