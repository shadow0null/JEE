from __future__ import annotations

FORMULAS = {
    'kinematics_velocity': {'subject':'physics','chapter':'Kinematics','formula':'v = u + at','latex':r'v=u+at','variables':['v','u','a','t']},
    'kinematics_displacement': {'subject':'physics','chapter':'Kinematics','formula':'s = ut + 1/2 at^2','latex':r's=ut+\\frac{1}{2}at^2','variables':['s','u','a','t']},
    'kinematics_third': {'subject':'physics','chapter':'Kinematics','formula':'v^2 = u^2 + 2as','latex':r'v^2=u^2+2as','variables':['v','u','a','s']},
    'projectile_range': {'subject':'physics','chapter':'Kinematics','formula':'R = u^2 sin(2theta)/g','latex':r'R=\\frac{u^2\\sin(2\\theta)}{g}','variables':['R','u','theta','g']},
    'newton_second': {'subject':'physics','chapter':'Laws of Motion','formula':'F = ma','latex':r'F=ma','variables':['F','m','a']},
    'friction': {'subject':'physics','chapter':'Laws of Motion','formula':'f = mu N','latex':r'f=\\mu N','variables':['f','mu','N']},
    'centripetal_force': {'subject':'physics','chapter':'Circular Motion','formula':'F = mv^2/r','latex':r'F=\\frac{mv^2}{r}','variables':['F','m','v','r']},
    'torque': {'subject':'physics','chapter':'Rotational Motion','formula':'tau = r F sin(theta)','latex':r'\\tau=rF\\sin\\theta','variables':['tau','r','F','theta']},
    'angular_momentum': {'subject':'physics','chapter':'Rotational Motion','formula':'L = I omega','latex':r'L=I\\omega','variables':['L','I','omega']},
    'coulomb_force': {'subject':'physics','chapter':'Electrostatics','formula':'F = k q1 q2 / r^2','latex':r'F=k\\frac{q_1q_2}{r^2}','variables':['F','q1','q2','r']},
    'electric_field_point': {'subject':'physics','chapter':'Electrostatics','formula':'E = k q / r^2','latex':r'E=k\\frac{q}{r^2}','variables':['E','q','r']},
    'electric_potential_point': {'subject':'physics','chapter':'Electrostatics','formula':'V = k q / r','latex':r'V=k\\frac{q}{r}','variables':['V','q','r']},
    'capacitance_parallel_plate': {'subject':'physics','chapter':'Capacitance','formula':'C = eps0 A / d','latex':r'C=\\frac{\\varepsilon_0A}{d}','variables':['C','A','d']},
    'ohm': {'subject':'physics','chapter':'Current Electricity','formula':'V = IR','latex':r'V=IR','variables':['V','I','R']},
    'magnetic_force_charge': {'subject':'physics','chapter':'Magnetism','formula':'F = q v B sin(theta)','latex':r'F=qvB\\sin\\theta','variables':['F','q','v','B','theta']},
    'faraday': {'subject':'physics','chapter':'EMI','formula':'emf = -dPhi/dt','latex':r'\\mathcal{E}=-\\frac{d\\Phi}{dt}','variables':['emf','flux_change','time']},
    'lens_formula': {'subject':'physics','chapter':'Ray Optics','formula':'1/f = 1/v - 1/u','latex':r'\\frac1f=\\frac1v-\\frac1u','variables':['f','v','u']},
    'photoelectric': {'subject':'physics','chapter':'Modern Physics','formula':'Kmax = h nu - phi','latex':r'K_{max}=h\\nu-\\phi','variables':['Kmax','nu','phi']},
}

def list_formulas(subject: str | None = None, chapter: str | None = None):
    rows=[]
    for key, item in FORMULAS.items():
        if subject and item['subject'].lower()!=subject.lower(): continue
        if chapter and chapter.lower() not in item['chapter'].lower(): continue
        rows.append({'id':key, **item})
    return rows

def get_formula(formula_id: str):
    item=FORMULAS.get(formula_id)
    return {'success':False,'error':'Unknown formula.'} if not item else {'success':True,'id':formula_id,**item}
