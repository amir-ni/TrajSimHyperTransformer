import math
import numpy as np

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def frechet_distance(trajA, trajB):
    nA, nB = len(trajA), len(trajB)
    ca = [[-1]*nB for _ in range(nA)]
    def _c(i,j):
        if ca[i][j] > -1:
            return ca[i][j]
        dist = euclidean(trajA[i], trajB[j])
        if i==0 and j==0:
            ca[i][j] = dist
        elif i>0 and j==0:
            ca[i][j] = max(_c(i-1,0), dist)
        elif i==0 and j>0:
            ca[i][j] = max(_c(0,j-1), dist)
        else:
            ca[i][j] = max(min(_c(i-1,j), _c(i-1,j-1), _c(i,j-1)), dist)
        return ca[i][j]
    return _c(nA-1,nB-1)

def dita_distance(trajA, trajB):
    arrA, arrB = np.array(trajA), np.array(trajB)
    minA, maxA = arrA.min(axis=0), arrA.max(axis=0)
    minB, maxB = arrB.min(axis=0), arrB.max(axis=0)
    return euclidean(minA, minB) + euclidean(maxA, maxB)

def tp_distance(trajA, trajB):
    def segment_centroids(traj, seg=5):
        step = max(1, len(traj)//seg)
        c = []
        idx = 0
        while idx < len(traj):
            chunk = traj[idx:idx+step]
            x = np.mean([p[0] for p in chunk])
            y = np.mean([p[1] for p in chunk])
            c.append((x,y))
            idx+=step
        return c
    cA = segment_centroids(trajA)
    cB = segment_centroids(trajB)
    L = min(len(cA), len(cB))
    total=0.0
    for i in range(L):
        total+=euclidean(cA[i], cB[i])
    return total/(L+1e-9)

def lcrs_distance(trajA, trajB):
    nA, nB = len(trajA), len(trajB)
    dp=[[0]*(nB+1) for _ in range(nA+1)]
    for i in range(1,nA+1):
        for j in range(1,nB+1):
            if trajA[i-1]==trajB[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])
    l=dp[nA][nB]
    return nA+nB-2*l

def approximate_eval(index, embs, ids, anchor_pos, k=5):
    """
    anchor_pos: dict: anchor_id -> list of positive ids
    We'll approximate recall@k, precision@k by searching in the index.
    """
    anchor_np = embs.numpy().astype(np.float32)
    D,I = index.search(anchor_np, k)
    total= len(ids)
    correct_recall=0
    correct_precision=0
    # recall => how many real pos are found
    # precision => how many retrieved are real pos
    for i,aid in enumerate(ids):
        rank_ids = I[i]
        positives = set(anchor_pos.get(aid,[]))
        if len(positives)==0:
            continue
        # recall: if any pos in top-k => success => we do binary
        # or we can do fraction
        found=0
        for rid in rank_ids:
            if rid in positives:
                found+=1
        # recall
        rec = found / len(positives)
        correct_recall+= rec
        # precision
        prec = found / k
        correct_precision+= prec
    # average
    recall_5 = correct_recall / total
    prec_5 = correct_precision / total
    return recall_5, prec_5