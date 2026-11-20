"""E2: exact connected-component sparse assignment."""
import numpy as np
from .accuracy_common import validate,candidate_lists,component_stats,expand_until_feasible

def e2_cc(ideal_positions, legal_sites, pitch_um, **kwargs):
    ideal, sites = validate(ideal_positions, legal_sites); k=int(kwargs.get("k",576)); cand=candidate_lists(ideal,sites,k); stats=kwargs.get("stats")
    comps=component_stats(ideal,sites,cand)
    if stats is not None:
        vals=[x[0] for x in comps]; stats.clear(); stats.update({"initial_k":k,"component_count":len(vals),"component_min":min(vals),"component_median":float(np.median(vals)),"component_max":max(vals)})
    return expand_until_feasible(ideal,sites,cand,stats,k,kwargs.get("max_k"))
