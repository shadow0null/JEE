import io, os, unittest
os.environ.setdefault('JEE_REQUIRE_API_KEY','0')
from fastapi.testclient import TestClient
from app import app

class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.c=TestClient(app)
    def test_health(self):
        r=self.c.get('/health'); self.assertEqual(r.status_code,200); self.assertEqual(r.json()['version'],'2.0.0')
    def test_capabilities(self):
        self.assertTrue(self.c.get('/capabilities').json()['capabilities']['verification'])
    def test_solve(self):
        r=self.c.post('/api/v2/solve',json={'question':'solve x^2 - 4 = 0'}); self.assertEqual(r.status_code,200); self.assertTrue(r.json()['success'])
    def test_multipart_pdf_invalid(self):
        r=self.c.post('/api/jee/pdf',files={'file':('bad.pdf',b'not pdf','application/pdf')}); self.assertIn(r.status_code,(200,422))
    def test_verify(self):
        r=self.c.post('/api/v2/verify',json={'expected':'x^2-1','student':'(x-1)*(x+1)','kind':'symbolic'}); self.assertTrue(r.json()['verified'])

if __name__=='__main__': unittest.main()
