"""E3: Hall-deficiency-directed candidate expansion."""
import numpy as np
from scipy.sparse.csgraph import maximum_bipartite_matching
from .accuracy_common import validate,candidate_lists,edge_graph,expand_until_feasible

def e3_hall(ideal_positions, legal_sites, pitch_um, **kwargs):
    ideal, sites=validate(ideal_positions,legal_sites); initial=int(kwargs.get("k",576)); k=initial; cand=candidate_lists(ideal,sites,k); stats=kwargs.get("stats"); expansions=affected=0; max_k=min(len(sites),int(kwargs.get("max_k",max(k*8,4096))))
    while True:
        match=maximum_bipartite_matching(edge_graph(ideal,sites,cand),perm_type="column"); missing=np.flatnonzero(match<0)
        if len(missing)==0 or k>=max_k: break
        affected += len(missing); expansions += 1; k=min(max_k,k*2); wider=candidate_lists(ideal,sites,k)
        for row in missing: cand[int(row)]=wider[int(row)]
    if stats is not None: stats.clear(); stats.update({"initial_k":initial,"final_k":k,"hall_expansions":expansions,"hall_affected_rows":affected})
    return expand_until_feasible(ideal,sites,cand,stats,initial,max_k)
