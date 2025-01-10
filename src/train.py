import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np

from dataset import TrajectoryDataset, collate_fn
from data_preprocessing import load_synthetic_data, convert_to_dataframe
from model import MultiLevelHyperTransformer
from losses import CircleLoss, TripletLoss, SegmentAlignmentLoss
from negative_sampling import build_faiss_index, find_hard_negatives
from metrics import approximate_eval

def run_train(config, logger):
    device = config['general']['device']
    # 1) Build data
    data = load_synthetic_data(
        num_trajs=config['data']['synthetic_num_trajs'],
        max_len=config['data']['synthetic_max_len']
    )
    df = convert_to_dataframe(data)
    dataset = TrajectoryDataset(df)
    loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=False, collate_fn=collate_fn)
    N = len(df)

    # 2) Model
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

    # 3) Losses
    loss_cfg = config['loss']
    circle_loss = CircleLoss(margin=loss_cfg['circle_margin'], gamma=loss_cfg['circle_gamma']).to(device)
    triplet_loss = TripletLoss(margin=loss_cfg['triplet_margin']).to(device)
    segalign_loss = SegmentAlignmentLoss(margin=loss_cfg['seg_margin'], gamma=loss_cfg['seg_gamma']).to(device)

    # 4) optimizer
    epochs = config['train']['epochs']
    lr = config['train']['learning_rate']
    wd = config['train']['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # 5) checkpoint dir
    ckpt_dir = config['train']['checkpoint_dir']
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    logger.info(f"Starting training for {epochs} epochs with dataset size={N}.")

    best_loss=1e9
    for ep in range(epochs):
        # Build embeddings for negative sampling
        all_embs=[]
        all_ids=[]
        model.eval()
        with torch.no_grad():
            for batch in loader:
                tids, points_list = batch
                B=len(tids)
                maxL = max([p.size(0) for p in points_list])
                padded = torch.zeros(B,maxL,3, device=device)
                for i,p in enumerate(points_list):
                    padded[i,:p.size(0),:]=p.to(device)
                embs = model(padded) # (B,d_out)
                all_embs.append(embs.cpu())
                all_ids.extend(tids)
        all_embs = torch.cat(all_embs, dim=0)  # (N,d_out)
        # build index
        index = build_faiss_index(all_embs.numpy().astype(np.float32))

        # anchor->pos for demonstration => random
        anchor_pos = {}
        for i,aid in enumerate(all_ids):
            # 1/5 chance we have a "pos" from i-1
            posset=[]
            if i>0 and i%5==0:
                posset.append(all_ids[i-1])
            anchor_pos[aid]=posset
        
        # find hard neg
        neg_dict = find_hard_negatives(index, all_embs, all_ids, anchor_pos, k=config['train']['faiss_k'])

        model.train()
        epoch_loss=0.0
        step=0
        for batch in loader:
            tids, points_list = batch
            B=len(tids)
            # create a small batch of anchor emb
            with torch.no_grad():
                maxL = max([p.size(0) for p in points_list])
                paddedA = torch.zeros(B,maxL,3, device=device)
                for i,p in enumerate(points_list):
                    paddedA[i,:p.size(0),:]=p.to(device)
                anchor_embs = model(paddedA)

            # build anchor->pos->neg
            circle_labels=[]
            trip_anchor=[]
            trip_pos=[]
            trip_neg=[]
            # segment alignment skip for demonstration
            seg_loss_total=0
            for i,aid in enumerate(tids):
                # pos
                posset=anchor_pos.get(aid,[])
                label=1 if len(posset)>0 else 0
                circle_labels.append(label)
                if label==1:
                    p_id=posset[0]
                else:
                    p_id=None
                # neg
                negset=neg_dict.get(aid,[])
                n_id=negset[0] if len(negset)>0 else None
                trip_anchor.append(anchor_embs[i,:]) 
                if p_id is not None:
                    idx_p=all_ids.index(p_id)
                    trip_pos.append(torch.tensor(all_embs[idx_p,:],device=device))
                else:
                    trip_pos.append(torch.zeros_like(anchor_embs[i,:]))
                if n_id is not None:
                    idx_n=all_ids.index(n_id)
                    trip_neg.append(torch.tensor(all_embs[idx_n,:],device=device))
                else:
                    trip_neg.append(torch.zeros_like(anchor_embs[i,:]))
            
            circle_label_tensor = torch.tensor(circle_labels, device=device,dtype=torch.float)
            anchor_batch = anchor_embs

            trip_anchor = torch.stack(trip_anchor,dim=0) # (B,d)
            trip_pos = torch.stack(trip_pos, dim=0)
            trip_neg = torch.stack(trip_neg, dim=0)

            # forward pass for segments alignment is omitted, naive approach
            # we do partial seg alignment => 0
            seg_loss_val = torch.tensor(0.0, device=device)

            # forward pass is partial => we do not re-run the entire model for local seg alignment
            # but in real usage, you'd parse sub-route embeddings from the gating steps
            # and match them. We'll skip for demonstration.

            loss_c = circle_loss(anchor_batch, anchor_batch, circle_label_tensor)
            loss_t = triplet_loss(trip_anchor, trip_pos, trip_neg)
            loss_s = seg_loss_val

            total_loss = config['loss']['lambda_circle']*loss_c + \
                         config['loss']['lambda_triplet']*loss_t + \
                         config['loss']['lambda_seg']*loss_s
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss+= total_loss.item()
            step+=1
        avg_loss = epoch_loss/step
        logger.info(f"Epoch={ep+1}, avg_loss={avg_loss:.4f}")

        # approximate eval each epoch
        if (ep+1)%config['train']['eval_interval']==0:
            # do approximate evaluation
            test_recall, test_precision = approximate_eval(index, all_embs, all_ids, anchor_pos, k=5)
            logger.info(f"[Approx Eval] Recall@5={test_recall:.4f}, Precision@5={test_precision:.4f}")

        # save checkpoint
        if config['train']['save_checkpoint']:
            if avg_loss<best_loss:
                best_loss=avg_loss
                path_ckpt = os.path.join(ckpt_dir, "best_model.pth")
                torch.save(model.state_dict(), path_ckpt)
                logger.info(f"Saved best model with avg_loss={avg_loss:.4f} to {path_ckpt}")

    logger.info("Training complete.")
