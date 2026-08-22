from __future__ import annotations
from typing import List, Dict, Any
import re

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE=True
except Exception:
    SKLEARN_AVAILABLE=False


def search_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int=5) -> dict:
    if not isinstance(query,str) or not query.strip(): return {'success':False,'error':'query is required.'}
    if not isinstance(chunks,list) or not chunks: return {'success':False,'error':'chunks are required.'}
    top_k=max(1,min(int(top_k),10))
    docs=[str(c.get('text',''))[:12000] for c in chunks]
    if SKLEARN_AVAILABLE:
        try:
            vec=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=25000)
            mat=vec.fit_transform(docs+[query])
            sims=cosine_similarity(mat[-1],mat[:-1]).ravel()
            ranked=sorted(enumerate(sims),key=lambda x:x[1],reverse=True)[:top_k]
            return {'success':True,'method':'tfidf','results':[{'score':round(float(s),5), **chunks[i]} for i,s in ranked if s>0]}
        except Exception:
            pass
    terms=set(re.findall(r'[A-Za-z0-9]{3,}',query.lower()))
    scored=[]
    for i,text in enumerate(docs):
        low=text.lower(); score=sum(low.count(t) for t in terms)
        if score: scored.append((i,score))
    scored.sort(key=lambda x:x[1],reverse=True)
    return {'success':True,'method':'keyword','results':[{'score':float(s), **chunks[i]} for i,s in scored[:top_k]]}
