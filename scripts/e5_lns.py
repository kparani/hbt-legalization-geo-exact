"""E5: deterministic excess-displacement large-neighborhood repair."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .baselines import sliding_window

def e5_lns(ideal_positions, legal_sites, pitch_um, **kwargs):
    ideal=np.asarray(ideal_positions,float); sites=np.asarray(legal_sites,float); assignment,_=sliding_window(ideal,sites,pitch_um=pitch_um); tree=cKDTree(sites); owner=np.full(len(sites),-1,int); owner[assignment]=np.arange(len(ideal)); repairs=0; rounds=int(kwargs.get("rounds",3))
    for _ in range(rounds):
        disp=np.abs(ideal-sites[assignment]).sum(1); worst=np.argsort(-disp,kind="stable")[:max(1,int(.05*len(ideal)))]
        for row in worst:
            rows=np.flatnonzero(np.linalg.norm(ideal-ideal[row],axis=1)<=90.*pitch_um)
            # Cap the exact neighborhood at 128 rows to keep each repeated
            # dense LAP bounded while retaining a generous geometric radius.
            if len(rows)>128: rows=rows[np.argsort(np.linalg.norm(ideal[rows]-ideal[row],axis=1),kind="stable")[:128]]
            _,cand=tree.query(ideal[row],k=min(len(sites),max(256,8*len(rows))),workers=1); cand=np.unique(np.atleast_1d(cand)); cand=cand[(owner[cand]<0)|np.isin(owner[cand],rows)]
            if len(cand)<len(rows): continue
            old=np.abs(ideal[rows]-sites[assignment[rows]]).sum(); cost=np.abs(ideal[rows,None]-sites[cand][None,:]).sum(2); rr,cc=linear_sum_assignment(cost); proposed=np.full(len(rows),-1,int); proposed[rr]=cand[cc]; new=np.abs(ideal[rows]-sites[proposed]).sum()
            if new<old-1e-9:
                for r in rows: owner[assignment[r]]=-1
                for r,s in zip(rows,proposed): assignment[r]=s; owner[s]=r
                repairs+=1
    if kwargs.get("stats") is not None: kwargs["stats"].update({"repair_rounds":rounds,"repairs":repairs})
    return assignment,float(np.abs(ideal-sites[assignment]).sum())
