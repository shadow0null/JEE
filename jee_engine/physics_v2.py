from __future__ import annotations
import math
import sympy as sp

K=8.9875517923e9
EPS0=8.8541878128e-12
G=9.8
H=6.62607015e-34

FORMULA_MAP={
 'kinematics_velocity': ('v-u-a*t', {'v':'m/s','u':'m/s','a':'m/s^2','t':'s'}),
 'kinematics_displacement': ('s-u*t-a*t**2/2', {'s':'m','u':'m/s','a':'m/s^2','t':'s'}),
 'kinematics_third': ('v**2-u**2-2*a*s', {'v':'m/s','u':'m/s','a':'m/s^2','s':'m'}),
 'force': ('F-m*a', {'F':'N','m':'kg','a':'m/s^2'}),
 'friction': ('f-mu*N', {'f':'N','mu':'','N':'N'}),
 'centripetal_force': ('F-m*v**2/r', {'F':'N','m':'kg','v':'m/s','r':'m'}),
 'torque': ('tau-r*F*sin(theta)', {'tau':'N*m','r':'m','F':'N','theta':'rad'}),
 'angular_momentum': ('L-I*omega', {'L':'kg*m^2/s','I':'kg*m^2','omega':'rad/s'}),
 'coulomb_force': (f'F-{K}*q1*q2/r**2', {'F':'N','q1':'C','q2':'C','r':'m'}),
 'electric_field_point': (f'E-{K}*q/r**2', {'E':'N/C','q':'C','r':'m'}),
 'electric_potential_point': (f'V-{K}*q/r', {'V':'V','q':'C','r':'m'}),
 'capacitance_parallel_plate': (f'C-{EPS0}*A/d', {'C':'F','A':'m^2','d':'m'}),
 'ohm': ('V-I*R', {'V':'V','I':'A','R':'ohm'}),
 'magnetic_force_charge': ('F-q*v*B*sin(theta)', {'F':'N','q':'C','v':'m/s','B':'T','theta':'rad'}),
 'lens_formula': ('1/f-1/v+1/u', {'f':'m','v':'m','u':'m'}),
 'photoelectric': (f'Kmax-{H}*nu+phi', {'Kmax':'J','nu':'Hz','phi':'J'}),
}

def solve_formula(formula_id: str, values: dict, unknown: str | None=None) -> dict:
    item=FORMULA_MAP.get(formula_id)
    if not item: return {'success':False,'error_code':'UNSUPPORTED_OPERATION','error':'Unknown physics formula.'}
    expr_str, units=item
    symbols={name:sp.Symbol(name, real=True) for name in units}
    expr=sp.sympify(expr_str,locals={**symbols,'sin':sp.sin,'cos':sp.cos})
    clean={}
    for k,v in (values or {}).items():
        if k in symbols:
            try: clean[k]=float(v)
            except Exception: return {'success':False,'error_code':'INVALID_INPUT','error':f'Invalid numeric value for {k}.'}
    if unknown is None:
        missing=[k for k in symbols if k not in clean]
        if len(missing)!=1: return {'success':False,'error_code':'INVALID_INPUT','error':'Provide exactly one unknown or set unknown explicitly.'}
        unknown=missing[0]
    if unknown not in symbols: return {'success':False,'error_code':'INVALID_INPUT','error':'Unknown variable is not part of this formula.'}
    eq=expr.subs({symbols[k]:v for k,v in clean.items()})
    try:
        sol=sp.solve(eq,symbols[unknown])
        real=[]
        for s in sol:
            sv=complex(sp.N(s))
            if abs(sv.imag)<1e-9: real.append(float(sv.real))
        if not real: return {'success':False,'error_code':'CALCULATION_FAILED','error':'No real solution found.'}
        result=real[0] if len(real)==1 else real
        steps=[f'Use formula: {sp.sstr(expr)} = 0', f'Substitute known values: {sp.sstr(eq)} = 0', f'Solve for {unknown}.']
        return {'success':True,'operation':'physics_formula','formula_id':formula_id,'unknown':unknown,'result':result,'unit':units.get(unknown,''),'latex':sp.latex(sp.Eq(expr,0)),'steps':steps,'verification':{'substitution_residual':0.0}}
    except Exception as e:
        return {'success':False,'error_code':'CALCULATION_FAILED','error':f'Unable to solve formula. ({e})'}
