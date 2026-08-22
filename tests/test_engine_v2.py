import io, os, unittest
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from jee_engine import router, pdf_engine, math_engine
from jee_engine.physics_v2 import solve_formula
from jee_engine.verification_engine import verify_symbolic
from jee_engine.retrieval_engine import search_chunks

class EngineV2Tests(unittest.TestCase):
    def test_router_confidence(self):
        r=router.classify('Find projectile range and plot trajectory')
        self.assertGreater(r['confidence'],0.5)
        self.assertIn(r['type'], ('PHYSICS','GRAPH'))
        self.assertTrue(r['topic'])
    def test_reverse_formula(self):
        r=solve_formula('force',{'F':10,'m':2},'a')
        self.assertTrue(r['success']); self.assertAlmostEqual(r['result'],5.0)
    def test_symbolic_verify(self):
        r=verify_symbolic('x^2-1','(x-1)*(x+1)')
        self.assertTrue(r['success']); self.assertTrue(r['verified'])
    def test_extended_math(self):
        self.assertTrue(math_engine.arithmetic_progression(2,3,5)['success'])
        self.assertTrue(math_engine.binomial_coefficient(5,2)['success'])
    def test_tfidf(self):
        r=search_chunks('gauss electric flux',[{'text':'Gauss law relates electric flux to enclosed charge','page_start':1},{'text':'organic chemistry reaction','page_start':2}],1)
        self.assertTrue(r['success']); self.assertEqual(r['results'][0]['page_start'],1)

if __name__=='__main__': unittest.main()
