from __future__ import annotations
import math
from typing import Any
import sympy as sp
from .safety import time_limit, SafetyError


def _fail(msg, code='VERIFY_FAILED'):
    return {'success':False,'error_code':code,'error':msg}

def verify_symbolic(expected: str, student: str, var: str='x') -> dict:
    try:
        x=sp.Symbol(var)
        local={var:x}
        with time_limit():
            a=sp.sympify(expected.replace('^','**'), locals=local)
            b=sp.sympify(student.replace('^','**'), locals=local)
            equivalent=bool(sp.simplify(a-b)==0)
        return {'success':True,'verified':equivalent,'method':'symbolic_equivalence','expected_latex':sp.latex(a),'student_latex':sp.latex(b)}
    except Exception as e:
        return _fail(f'Could not verify symbolic answer safely. ({e})')

def verify_numeric(expected: Any, student: Any, rel_tol: float=1e-7, abs_tol: float=1e-9) -> dict:
    try:
        a=float(expected); b=float(student)
        return {'success':True,'verified':math.isclose(a,b,rel_tol=rel_tol,abs_tol=abs_tol),'method':'numeric_tolerance','expected':a,'student':b}
    except Exception as e:
        return _fail(f'Could not compare numeric values. ({e})')

def verify(payload: dict) -> dict:
    expected=payload.get('expected'); student=payload.get('student')
    if expected is None or student is None: return _fail('expected and student are required.','INVALID_INPUT')
    kind=str(payload.get('kind','auto')).lower()
    if kind=='numeric': return verify_numeric(expected,student)
    if kind=='symbolic': return verify_symbolic(str(expected),str(student),str(payload.get('var','x')))
    num=verify_numeric(expected,student)
    if num.get('success'):
        return num
    return verify_symbolic(str(expected),str(student),str(payload.get('var','x')))
