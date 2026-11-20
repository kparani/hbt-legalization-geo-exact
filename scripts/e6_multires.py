"""E6: exact multi-resolution sparse matching."""
import numpy as np
from .accuracy_common import validate,candidate_lists,solve_exact_sparse

def e6_multires(ideal_positions, legal_sites, pitch_um, **kwargs):
    ideal,sites=validate(ideal_positions,legal_sites); k=int(kwargs.get("k",4096)); cand=candidate_lists(ideal,sites,k); stats=kwargs.get("stats"); a,t=solve_exact_sparse(ideal,sites,cand,stats); costs=np.abs(ideal-sites[a]).sum(1); rows=np.flatnonzero(costs>=np.percentile(costs,90.))
    if len(rows):
        wide=candidate_lists(ideal,sites,min(len(sites),k*2))
        for row in rows: cand[int(row)]=wide[int(row)]
        a,t=solve_exact_sparse(ideal,sites,cand,stats)
    if stats is not None: stats.update({"initial_k":k,"refined_rows":int(len(rows)),"final_k":k*2 if len(rows) else k})
    return a,t
