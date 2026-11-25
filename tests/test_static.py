import ast
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class RepositoryTests(unittest.TestCase):
    def test_python_syntax(self):
        for name in ('app.py','insurance_agent.py'):
            ast.parse((ROOT/name).read_text())
    def test_policy_ids(self):
        ids=[p.read_text().splitlines()[0] for p in (ROOT/'data/policies').glob('*.txt')]
        self.assertEqual(len(ids),4)
        self.assertEqual(len(set(ids)),4)
    def test_credentials_template(self):
        self.assertTrue(all(line.endswith('=') for line in (ROOT/'.env.example').read_text().splitlines()))
    def test_retry_cap(self):
        tree=ast.parse((ROOT/'insurance_agent.py').read_text())
        caps=[n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAX_RETRIEVAL_RETRIES_CAP' for t in n.targets)]
        self.assertEqual(caps,[2])
if __name__=='__main__': unittest.main()
