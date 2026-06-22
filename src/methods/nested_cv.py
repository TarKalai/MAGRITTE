"""Leakage-free nested cross-validation and the repeated model comparison.

Outer StratifiedGroupKFold folds are scored once each; an inner group split supplies the
early-stopping set (model selection by validation PR-AUC). Reports threshold-independent
metrics only (ROC-AUC, PR-AUC) as mean +/- std over R repeated runs.
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.loaders import ACTIVE_FEATURES, build_fold_data
from src.methods.models import GINEFraudModel, LSTMFraudModel, TransformerFraudModel
from src.utils.training import _train_one_epoch, _predict_proba


def evaluate_nested_cv(model_fn, source_ids, labels_augmented, is_cut, session_deltas, all_cut_positions,
                       label_to_int, active_feature_columns, n_splits=5, n_inner_splits=4, n_seeds=3,
                       n_epochs=20, patience=5, lr=1e-3, weight_decay=1e-2, batch_size=32,
                       cut_only_training=True, seed_offset=0, device="cpu"):
    """Leakage-free nested CV. Returns threshold-independent metrics only (ROC-AUC, PR-AUC).

    Per outer fold the held-out group is scored once; an inner group-split provides the early-stopping
    set. Model selection uses validation PR-AUC (threshold-free). No decision threshold is ever applied.
    """
    device = torch.device(device)
    elig = [i for i, (s, l, cf) in enumerate(zip(source_ids, labels_augmented, is_cut))
            if not (cut_only_training and not cf)]
    sids = [source_ids[i] for i in elig]; labs = [labels_augmented[i] for i in elig]
    iscut = [is_cut[i] for i in elig]
    y = np.array([1 if l == "fraud" else 0 for l in labs]); sa = np.array(sids)
    net = max(label_to_int.values()) + 2
    oof_true, oof_prob = [], []
    for tr, te in StratifiedGroupKFold(n_splits=n_splits).split(sids, y, groups=sa):
        rt, rv = next(StratifiedGroupKFold(n_splits=n_inner_splits).split(tr, y[tr], groups=sa[tr]))
        in_tr, in_val = tr[rt], tr[rv]
        def _b(ix):
            return build_fold_data(ix, 1, sids, labs, iscut, session_deltas, all_cut_positions,
                                   label_to_int, active_feature_columns, None, None, None, False)
        trd, vd, td = _b(in_tr), _b(in_val), _b(te)
        if not trd or not vd or not td:
            continue
        yv = np.array([int(d.y.item()) for d in vd]); yt = np.array([int(d.y.item()) for d in td])
        if len(set(yv)) < 2 or len(set(yt)) < 2:
            continue
        npz = sum(1 for d in trd if d.y.item() == 1)
        pw = torch.tensor([max(len(trd) - npz, 1) / max(npz, 1)], dtype=torch.float32)
        nsc = trd[0].x_scalar.shape[-1]
        tl = DataLoader(trd, batch_size=batch_size, shuffle=True)
        vl = DataLoader(vd, batch_size=batch_size, shuffle=False)
        el = DataLoader(td, batch_size=batch_size, shuffle=False)
        seed_te = np.zeros(len(td))
        for seed in range(n_seeds):
            torch.manual_seed(seed + seed_offset); np.random.seed(seed + seed_offset)
            model = model_fn(net, nsc).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            best_ap, best_state, left = -1, None, patience
            for _ in range(n_epochs):
                _train_one_epoch(model, tl, opt, pw, device)
                ap = average_precision_score(yv, _predict_proba(model, vl, device))   # threshold-free selection
                if ap > best_ap:
                    best_ap = ap
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    left = patience
                else:
                    left -= 1
                    if left <= 0:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            seed_te += _predict_proba(model, el, device)
        seed_te /= n_seeds
        oof_true.extend(yt.tolist()); oof_prob.extend(seed_te.tolist())
    oof_true, oof_prob = np.array(oof_true), np.array(oof_prob)
    return {"roc_auc": roc_auc_score(oof_true, oof_prob),
            "pr_auc": average_precision_score(oof_true, oof_prob),
            "n_test": int(len(oof_true)), "n_fraud": int(oof_true.sum())}


def model_factories(event_emb_dim=16, hidden_dim=32, n_layers=2, dropout=0.0):
    return {
        "GINE": lambda net, nsc: GINEFraudModel(n_event_types=net, n_scalar_node_features=nsc,
                    n_edge_features=1, event_emb_dim=event_emb_dim, hidden_dim=hidden_dim,
                    n_layers=n_layers, dropout=dropout),
        "LSTM": lambda net, nsc: LSTMFraudModel(n_event_types=net, n_scalar_node_features=nsc,
                    event_emb_dim=event_emb_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout),
        "Transformer": lambda net, nsc: TransformerFraudModel(n_event_types=net, n_scalar_node_features=nsc,
                    event_emb_dim=event_emb_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout),
    }


def repeated_comparison(session_deltas, all_cut_positions, label_to_int, source_ids, labels_augmented,
                        is_cut, active_feature_columns=ACTIVE_FEATURES, R=3, model_kwargs=None, **kw):
    factories = model_factories(**(model_kwargs or {}))
    rows = []
    for name, fn in factories.items():
        rocs, prs = [], []
        for rep in range(R):
            r = evaluate_nested_cv(fn, source_ids, labels_augmented, is_cut, session_deltas,
                                   all_cut_positions, label_to_int, list(active_feature_columns),
                                   seed_offset=rep * 100, **kw)
            rocs.append(r["roc_auc"]); prs.append(r["pr_auc"])
            print(f"  {name} rep {rep + 1}/{R}: ROC-AUC={r['roc_auc']:.3f}  PR-AUC={r['pr_auc']:.3f}", flush=True)
        rows.append({"model": name,
                     "ROC_AUC_mean": np.mean(rocs), "ROC_AUC_std": np.std(rocs),
                     "PR_AUC_mean": np.mean(prs), "PR_AUC_std": np.std(prs)})
    return pd.DataFrame(rows)
