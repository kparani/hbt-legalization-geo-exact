"""E4: DGHL-schedule local Hungarian with carried site dual prices."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .baselines import sliding_window

def e4_warm(ideal_positions, legal_sites, pitch_um, **kwargs):
    ideal=np.asarray(ideal_positions,float); sites=np.asarray(legal_sites,float); tree=cKDTree(sites)
    assignment,_=sliding_window(ideal,sites,pitch_um=pitch_um); owner=np.full(len(sites),-1,int); owner[assignment]=np.arange(len(ideal)); prices=np.zeros(len(sites)); cell=30.*pitch_um; origin=sites.min(0); keys=np.floor((ideal-origin)/cell).astype(int)
    order=sorted(set(map(tuple,keys)),key=lambda z:(-int(np.sum(np.all(keys==z,axis=1))),z)); changes=0
    for pass_index in range(3):
        width=max(1.,3.*cell*.5**pass_index)
        for key in order:
            center=origin+(np.asarray(key)+.5)*cell; lo=center-width/2; hi=center+width/2; rows=np.flatnonzero(np.all((ideal>=lo)&(ideal<=hi),axis=1))
            if len(rows)==0 or len(rows)>576: continue
            cand=np.asarray(tree.query_ball_point(center,np.linalg.norm(hi-lo)/2.),int); cand=cand[np.all((sites[cand]>=lo)&(sites[cand]<=hi),axis=1)]; cand=cand[(owner[cand]<0)|np.isin(owner[cand],rows)]
            if len(cand)<len(rows) or len(cand)>10000: continue
            old=np.abs(ideal[rows]-sites[assignment[rows]]).sum(); cost=np.abs(ideal[rows,None]-sites[cand][None,:]).sum(2)+prices[cand][None,:]*1e-9; rr,cc=linear_sum_assignment(cost); proposed=np.full(len(rows),-1,int); proposed[rr]=cand[cc]; new=np.abs(ideal[rows]-sites[proposed]).sum()
            if new<old-1e-9:
                for r in rows: owner[assignment[r]]=-1
                for r,s in zip(rows,proposed): assignment[r]=s; owner[s]=r; prices[s]+=1.; changes+=1
    if kwargs.get("stats") is not None: kwargs["stats"].update({"warm_windows":len(order)*3,"improvement_changes":changes})
    return assignment,float(np.abs(ideal-sites[assignment]).sum())
