import torch
import os
import numpy as np

from dataset import TrajectoryDataset, collate_fn
from data_preprocessing import load_synthetic_data, convert_to_dataframe
from model import MultiLevelHyperTransformer
from metrics import approximate_eval
from negative_sampling import build_faiss_index
from torch.utils.data import DataLoader

def run_test(config, logger):
    device = config['general']['device']
    checkpoint_path = config['test']['checkpoint_path']

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    # load data
    data = load_synthetic_data(
        num_trajs=config['data']['synthetic_num_trajs'],
        max_len=config['data']['synthetic_max_len']
    )
    df = convert_to_dataframe(data)

    ds = TrajectoryDataset(df)
    loader = DataLoader(ds,batch_size=config['test']['batch_size'],shuffle=False,collate_fn=collate_fn)

    # load model
    model_cfg = config['model']
    model = MultiLevelHyperTransformer(
        d_out=model_cfg['d_out'],
        r_s=model_cfg['r_s'],
        r_t=model_cfg['r_t'],
        gating_threshold=model_cfg['gating_threshold'],
        seg_transformer_layers=model_cfg['seg_transformer_layers'],
        seg_transformer_heads=model_cfg['seg_transformer_heads'],
        global_transformer_layers=model_cfg['global_transformer_layers'],
        global_transformer_heads=model_cfg['global_transformer_heads']
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    model.eval()
    all_embs=[]
    all_ids=[]
    with torch.no_grad():
        for batch in loader:
            tids, points_list = batch
            maxL = max([p.size(0) for p in points_list])
            B=len(points_list)
            padded = torch.zeros(B,maxL,3,device=device)
            for i,p in enumerate(points_list):
                padded[i,:p.size(0),:]=p.to(device)
            embs = model(padded)
            all_embs.append(embs.cpu())
            all_ids.extend(tids)
    all_embs = torch.cat(all_embs,dim=0)
    index = build_faiss_index(all_embs.numpy().astype(np.float32))

    # fake pos for demonstration
    anchor_pos={}
    for i,aid in enumerate(all_ids):
        posset=[]
        if i>0 and i%5==0:
            posset.append(all_ids[i-1])
        anchor_pos[aid]=posset
    recall_5, prec_5 = approximate_eval(index, all_embs, all_ids, anchor_pos, k=5)
    logger.info(f"Test approximate metrics: Recall@5={recall_5:.4f}, Prec@5={prec_5:.4f}")

    logger.info("Test finished.")
