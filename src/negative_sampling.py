import faiss
import numpy as np

def build_faiss_index(embs):
    """
    embs: (N,d) numpy
    returns a Faiss index
    """
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index

def find_hard_negatives(index, anchor_embs, anchor_ids, pos_ids, k=10):
    """
    anchor_embs: (B,d)
    anchor_ids: list of ids
    pos_ids: dictionary anchor_id -> list of pos_ids
    returns dictionary anchor_id -> list of neg_ids
    """
    anchor_np = anchor_embs.cpu().numpy().astype(np.float32)
    D,I = index.search(anchor_np, k) # top-k
    # for each anchor, we pick from I ignoring pos_ids
    neg_dict={}
    for i,aid in enumerate(anchor_ids):
        # list of top neighbors
        rank_ids = I[i]  # top-k neighbor indices
        # skip self
        # skip pos
        posset = set(pos_ids.get(aid,[]))
        cands = []
        for rid in rank_ids:
            if rid!=aid and (rid not in posset):
                cands.append(rid)
        neg_dict[aid] = cands
    return neg_dict
