"""Low-level training / inference helpers shared by all models."""
import numpy as np
import torch
import torch.nn.functional as F


def _train_one_epoch(model, loader, optimizer, pos_weight, device):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch.y.view(-1), pos_weight=pos_weight.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0


@torch.no_grad()
def _predict_proba(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = batch.to(device)
        probs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])

