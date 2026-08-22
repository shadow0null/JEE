"""StudyDesk JEE Engine 2.0 HTTP API.
Deterministic JEE/NEET tools for the StudyDesk Tutor. No LLM/API calls occur here.
"""
from __future__ import annotations

import base64
import importlib.util
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import jee_main
from jee_engine import pdf_engine, router
from jee_engine.formula_registry import list_formulas, get_formula
from jee_engine.physics_v2 import solve_formula
from jee_engine.verification_engine import verify
from jee_engine.retrieval_engine import search_chunks

MAX_REQUEST_BODY_BYTES = 25 * 1024 * 1024
API_KEY = os.getenv('JEE_API_KEY', '').strip()
REQUIRE_KEY = os.getenv('JEE_REQUIRE_API_KEY', '0').strip().lower() in {'1','true','yes','on'}
CORS_ORIGINS = [x.strip() for x in os.getenv('JEE_CORS_ORIGINS','https://studydesk.fun').split(',') if x.strip()]

app = FastAPI(title='StudyDesk JEE Engine', version='2.0.0', docs_url='/docs', redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=False,
                   allow_methods=['GET','POST','OPTIONS'], allow_headers=['Content-Type','X-API-Key'])

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=600)

class StructuredRequest(BaseModel):
    type: Optional[str] = None
    operation: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

class VerifyRequest(BaseModel):
    expected: Any
    student: Any
    kind: str = 'auto'
    var: str = 'x'

class PhysicsFormulaRequest(BaseModel):
    formula_id: str
    values: Dict[str, Any] = Field(default_factory=dict)
    unknown: Optional[str] = None

class SearchChunksRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    chunks: List[Dict[str, Any]]
    top_k: int = 5

@app.middleware('http')
async def request_guard(request: Request, call_next):
    started=time.perf_counter()
    content_length=request.headers.get('content-length')
    if content_length:
        try:
            if int(content_length)>MAX_REQUEST_BODY_BYTES:
                return JSONResponse(status_code=413,content={'success':False,'error_code':'PAYLOAD_TOO_LARGE','error':'Request body is too large.'})
        except ValueError:
            pass
    try:
        response=await call_next(request)
        response.headers['X-StudyDesk-Engine']='2.0.0'
        response.headers['X-StudyDesk-Latency-Ms']=str(int((time.perf_counter()-started)*1000))
        return response
    except Exception as exc:
        return JSONResponse(status_code=500,content={'success':False,'error_code':'INTERNAL_ERROR','error':f'{type(exc).__name__}: {exc}'})

def _check_key(x_api_key: Optional[str]) -> None:
    if REQUIRE_KEY and not API_KEY:
        raise HTTPException(status_code=503,detail='JEE_API_KEY is required but not configured on the engine.')
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401,detail='Unauthorized.')

def _json_safe_result(result: dict) -> dict:
    result=dict(result or {})
    if result.get('success') and result.get('type')=='GRAPH':
        path=result.pop('file_path',None)
        if path:
            try:
                with open(path,'rb') as fh: result['image_base64']=base64.b64encode(fh.read()).decode('ascii')
            finally:
                try: Path(path).unlink()
                except OSError: pass
    return result

def _normalize_result(result: dict, classification: Optional[dict]=None) -> dict:
    r=_json_safe_result(result or {})
    r.setdefault('success',False)
    if classification:
        r.setdefault('route',classification.get('type'))
        r.setdefault('topic',classification.get('topic'))
        r.setdefault('route_confidence',classification.get('confidence'))
        r.setdefault('secondary_routes',classification.get('secondary',[]))
    if r.get('success'):
        r.setdefault('error_code',None)
        if 'result' in r and 'final_answer' not in r:
            r['final_answer']=r.get('result_str',r.get('result'))
        r.setdefault('steps',[])
    else:
        r.setdefault('error_code','UNSUPPORTED_OPERATION')
        r.setdefault('fallback','external_ai')
    return r

def _dep(name:str)->bool:
    return importlib.util.find_spec(name) is not None

@app.get('/')
def root():
    return {'service':'StudyDesk JEE Engine','version':'2.0.0','status':'ok','docs':'/docs'}

@app.get('/health')
def health():
    deps={'sympy':_dep('sympy'),'numpy':_dep('numpy'),'scipy':_dep('scipy'),'pint':_dep('pint'),
          'pymupdf':_dep('fitz'),'matplotlib':_dep('matplotlib'),'sklearn':_dep('sklearn'),
          'networkx':_dep('networkx'),'pillow':_dep('PIL'),'python_multipart':_dep('multipart'),
          'ocr_python':_dep('pytesseract'),'tesseract_binary':bool(shutil.which('tesseract'))}
    critical=all(deps[k] for k in ['sympy','numpy','scipy','pint','pymupdf','python_multipart'])
    return {'status':'ok' if critical else 'degraded','service':'local-jee-engine','version':'2.0.0',
            'python':platform.python_version(),'dependencies':deps,
            'auth':{'key_configured':bool(API_KEY),'required':REQUIRE_KEY,'mode':'required' if REQUIRE_KEY else ('protected-if-key-sent' if API_KEY else 'open-warning')},
            'cors_origins':CORS_ORIGINS}

@app.get('/capabilities')
def capabilities():
    h=health()
    return {'success':True,'version':'2.0.0','capabilities':{
        'math':True,'physics':True,'chemistry':True,'biology':True,'units':True,'graphs':True,
        'pdf':bool(h['dependencies']['pymupdf']),'pdf_multipart':bool(h['dependencies']['python_multipart']),
        'ocr':bool(h['dependencies']['tesseract_binary']),'tfidf_retrieval':bool(h['dependencies']['sklearn']),
        'verification':True,'formula_registry':True,'reverse_physics_formula_solving':True,
        'router_confidence':True,'structured_steps_latex':True}}

@app.post('/api/v2/solve')
def solve(req: QuestionRequest, x_api_key: Optional[str]=Header(default=None)):
    _check_key(x_api_key)
    cls=router.classify(req.question)
    result=jee_main.process_text(req.question)
    return _normalize_result(result,cls)

# Backward compatible Tutor endpoint.
@app.post('/api/jee/question')
def question(req: QuestionRequest, x_api_key: Optional[str]=Header(default=None)):
    return solve(req,x_api_key)

@app.post('/api/jee/json')
def structured(payload: Dict[str,Any], x_api_key: Optional[str]=Header(default=None)):
    _check_key(x_api_key)
    try: return _normalize_result(jee_main.process_json(payload))
    except Exception as exc: return {'success':False,'error_code':'CALCULATION_FAILED','error':f'Local calculation unavailable. ({exc})','fallback':'external_ai'}

@app.post('/api/v2/physics/formula')
def physics_formula(req: PhysicsFormulaRequest, x_api_key: Optional[str]=Header(default=None)):
    _check_key(x_api_key); return solve_formula(req.formula_id,req.values,req.unknown)

@app.get('/api/v2/formulas')
def formulas(subject: Optional[str]=None, chapter: Optional[str]=None):
    return {'success':True,'formulas':list_formulas(subject,chapter)}

@app.get('/api/v2/formulas/{formula_id}')
def formula(formula_id:str): return get_formula(formula_id)

@app.post('/api/v2/verify')
def verify_answer(req: VerifyRequest, x_api_key: Optional[str]=Header(default=None)):
    _check_key(x_api_key); return verify(req.model_dump())

@app.post('/api/v2/pdf/search')
def pdf_search(req: SearchChunksRequest, x_api_key: Optional[str]=Header(default=None)):
    _check_key(x_api_key); return search_chunks(req.query,req.chunks,req.top_k)

async def _read_pdf_upload(file: UploadFile) -> bytes:
    if not file.filename: raise HTTPException(status_code=400,detail='PDF file is required.')
    raw=await file.read(MAX_REQUEST_BODY_BYTES+1)
    if len(raw)>MAX_REQUEST_BODY_BYTES: raise HTTPException(status_code=413,detail='PDF is too large.')
    return raw

# FIXED: real multipart endpoint used by StudyDesk PHP and curl -F file=@x.pdf
@app.post('/api/jee/pdf')
@app.post('/api/v2/pdf/upload')
async def pdf_upload(file: UploadFile=File(...), x_api_key: Optional[str]=Header(default=None)):
    _check_key(x_api_key)
    raw=await _read_pdf_upload(file)
    data=pdf_engine.extract_and_chunk(raw,use_ocr=True)
    if not data.get('success'):
        return JSONResponse(status_code=422,content={'ok':False,**data})
    # StudyDesk compatibility: ok + pages/chars are expected by tutor-documents.php
    return {'ok':True,**data}

# Backward-compatible JSON/base64 endpoint retained explicitly.
@app.post('/api/jee/pdf-base64')
def pdf_base64(payload: Dict[str,Any], x_api_key: Optional[str]=Header(default=None)):
    _check_key(x_api_key)
    data=payload.get('data_base64')
    if not isinstance(data,str) or not data: raise HTTPException(status_code=400,detail='PDF data is required.')
    try: raw=base64.b64decode(data,validate=True)
    except Exception: raise HTTPException(status_code=400,detail='Invalid PDF payload.')
    out=pdf_engine.extract_and_chunk(raw,use_ocr=True)
    return {'ok':bool(out.get('success')),**out}
