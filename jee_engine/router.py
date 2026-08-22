from __future__ import annotations
import re
from typing import Dict, List

CATEGORY_MATH='MATH'; CATEGORY_NUMERICAL='NUMERICAL'; CATEGORY_PHYSICS='PHYSICS'; CATEGORY_CHEMISTRY='CHEMISTRY'; CATEGORY_BIOLOGY='BIOLOGY'; CATEGORY_UNIT='UNIT'; CATEGORY_GRAPH='GRAPH'; CATEGORY_UNKNOWN='UNKNOWN'

PATTERNS={
 CATEGORY_GRAPH:[r'\bplot\b',r'\bgraph\b',r'\btrajectory\b',r'position[- ]time',r'velocity[- ]time',r'acceleration[- ]time'],
 CATEGORY_UNIT:[r'\bconvert\b',r'\bto\s+(?:m/s|km/h|kg|joule|newton|watt|ohm|volt|cm|mm|hour|hr)\b',r'->|→'],
 CATEGORY_PHYSICS:[r'\bforce\b',r'\bvelocity\b',r'\bacceleration\b',r'\bprojectile\b',r'\bfriction\b',r'\btorque\b',r'\bangular momentum\b',r'\bcoulomb\b',r'\belectric field\b',r'\belectric potential\b',r'\bcapacit',r'\bohm',r'\bcurrent electricity\b',r'\bmagnetic\b',r'\bfaraday\b',r'\binduct',r'\blens\b',r'\bmirror\b',r'\bphotoelectric\b',r'\bgravitation',r'\bshm\b'],
 CATEGORY_CHEMISTRY:[r'\bmolar',r'\bmole\b',r'\bstoichi',r'\bph\b',r'\bpoh\b',r'\bnernst\b',r'\belectrochem',r'\benthalpy\b',r'\bchemical kinetics\b',r'\bcalorimetry\b'],
 CATEGORY_BIOLOGY:[r'\bdna\b',r'\brna\b',r'\bgenotype\b',r'\bphenotype\b',r'\bhardy[- ]weinberg\b',r'\bpunnett\b',r'\bcodon\b',r'\bpopulation growth\b'],
 CATEGORY_NUMERICAL:[r'\bdot product\b',r'\bcross product\b',r'\bvector\b',r'\bnumerical integration\b',r'\binterpolat',r'\busing scipy\b',r'\busing numpy\b'],
 CATEGORY_MATH:[r'\bcalculate\b',r'\bsolve\b',r'\bdifferentiat',r'\bderivative\b',r'\bintegrat',r'\blimit\b',r'\bfactor\b',r'\bexpand\b',r'\bsimplify\b',r'\bdeterminant\b',r'\bmatrix\b',r'\bcomplex number',r'\bprobability\b',r'\bsequence\b',r'\bseries\b',r'\bparabola\b',r'\bellipse\b',r'\bhyperbola\b'],
}

TOPIC_RULES=[
 ('projectile_motion',[r'projectile',r'range',r'time of flight',r'maximum height']),
 ('kinematics',[r'velocity',r'acceleration',r'displacement',r'kinematics']),
 ('rotational_motion',[r'torque',r'angular momentum',r'rolling',r'moment of inertia']),
 ('electrostatics',[r'coulomb',r'electric field',r'electric potential',r'gauss']),
 ('capacitance',[r'capacitor',r'capacitance']),
 ('current_electricity',[r'ohm',r'current electricity',r'kirchhoff',r'wheatstone']),
 ('magnetism',[r'magnetic',r'lorentz',r'biot',r'ampere']),
 ('emi_ac',[r'faraday',r'induct',r'\bac\b',r'impedance',r'resonance']),
 ('optics',[r'lens',r'mirror',r'prism',r'interference',r'diffraction']),
 ('modern_physics',[r'photoelectric',r'bohr',r'radioactive',r'semiconductor']),
 ('calculus',[r'derivative',r'differentiat',r'integrat',r'limit']),
 ('algebra',[r'quadratic',r'complex',r'sequence',r'series',r'binomial']),
 ('coordinate_geometry',[r'parabola',r'ellipse',r'hyperbola',r'straight line',r'circle']),
]

def classify(question:str)->Dict:
    if not isinstance(question,str) or not question.strip(): return {'type':CATEGORY_UNKNOWN,'raw':question,'confidence':0.0,'topic':None,'secondary':[],'matched':[],'reason':'Empty input.'}
    text=question.strip(); lower=text.lower(); scores={k:0 for k in PATTERNS}; matched={k:[] for k in PATTERNS}
    for cat, pats in PATTERNS.items():
        for p in pats:
            if re.search(p,lower,re.I): scores[cat]+=1; matched[cat].append(p)
    if re.search(r'\d+\s*[a-zA-Z/]+\s*(?:to|→|->)\s*[a-zA-Z/]+',lower): scores[CATEGORY_UNIT]+=3
    if '=' in text and re.search(r'[A-Za-z]',text): scores[CATEGORY_MATH]+=1
    ranked=sorted(scores.items(),key=lambda kv:kv[1],reverse=True)
    best,best_score=ranked[0]
    if best_score<=0: return {'type':CATEGORY_UNKNOWN,'raw':text,'confidence':0.25,'topic':None,'secondary':[],'matched':[],'reason':'No supported deterministic pattern matched.'}
    # specificity bonuses
    if scores[CATEGORY_UNIT]>=3: best=CATEGORY_UNIT; best_score=scores[best]
    secondary=[c for c,s in ranked if c!=best and s>0][:3]
    total=max(1,sum(scores.values())); confidence=min(0.99,0.55+0.12*best_score+0.25*(best_score/total))
    topic=None
    for name,pats in TOPIC_RULES:
        if any(re.search(p,lower,re.I) for p in pats): topic=name; break
    return {'type':best,'raw':text,'confidence':round(confidence,3),'topic':topic,'secondary':secondary,'matched':matched.get(best,[])[:8]}
