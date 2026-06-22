"""Data loading and graph construction for the BPIC 2012 signature-fraud benchmark.

Builds per-session chain graphs (PyG `Data`) and the augmented training pool consumed by all
three models. `load_and_prepare` is the single entry point used by the comparison script.
"""
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data


# ---- benchmark configuration ----
COL_CASE, COL_EVENT, COL_TIME = "case_id", "activity", "timestamp"
SURROGATE_ACTIVITY = "O_ACCEPTED+COMPLETE"
DATA_PATH = "data/processed/bpic_2012_signature.parquet"
ACTIVE_FEATURES = ["cumulative_time", "event_frequency_rank"]


CUT_LEVEL = 1
DELTA_BINS = [0.03, 0.22, 1.11, 10]
DELTA_LABELS = ['INSTANT', 'FAST', 'MEDIUM', 'SLOW']


def compute_time_deltas(df):
    """Per-session list of (activity, delta_seconds); first event delta = 0."""
    session_deltas = {}
    for sid, group in df.groupby(COL_CASE):
        group = group.sort_values(COL_TIME)
        times = group[COL_TIME].tolist()
        labels = group[COL_EVENT].tolist()
        deltas = [0]
        for i in range(1, len(times)):
            deltas.append((times[i] - times[i - 1]).total_seconds())
        session_deltas[sid] = list(zip(labels, deltas))
    return session_deltas


def compute_cut_positions(session_deltas, surrogate_activity):
    """Cut position = index of the 1st/2nd/3rd occurrence of the surrogate transaction."""
    cuts = {}
    for sid, seq in session_deltas.items():
        positions = [i for i, (e, _) in enumerate(seq) if e == surrogate_activity]
        cuts[sid] = {n: p for n, p in enumerate(positions[:3], start=1) if p > 0}
    return cuts


def discretise_delta(delta: float):
    for th, label in zip(DELTA_BINS, DELTA_LABELS):
        if delta <= th:
            return label
    return DELTA_LABELS[-1]


def session_to_nx_original(session_id: str, session_deltas: dict, label_to_int: dict):
    """Build a directed graph for a session from the original sequence."""
    raw_seq_deltas = session_deltas[session_id]
    events = [event for event, _ in raw_seq_deltas]
    deltas = [delta for _, delta in raw_seq_deltas]

    edge_deltas = defaultdict(list)
    for i, (a, b) in enumerate(zip(events, events[1:])):
        if a not in label_to_int or b not in label_to_int:
            continue
        a_int = label_to_int[a]
        b_int = label_to_int[b]
        delta = deltas[i + 1] if i + 1 < len(deltas) else 0
        edge_deltas[(a_int, b_int)].append(delta)

    G = nx.DiGraph()
    for (a_int, b_int), delta_list in edge_deltas.items():
        mean_delta = float(np.mean(delta_list))
        G.add_edge(a_int, b_int, weight=len(delta_list), delta_bin=discretise_delta(mean_delta))
    return G


def build_augmented_training_graphs(valid_ids, session_deltas, all_cut_positions, labels, label_to_int, wl_iterations, cut_level):
    """Build augmented training graphs: complete sessions + cut sessions side by side."""
    graphs_augmented = []
    labels_augmented = []
    source_ids = []
    is_cut = []

    for sid in valid_ids:
        temp_deltas = {sid: session_deltas[sid]}
        G_complete = session_to_nx_original(sid, temp_deltas, label_to_int)

        if G_complete.number_of_edges() > 0:
            graphs_augmented.append(G_complete)
            labels_augmented.append(labels[sid])
            source_ids.append(sid)
            is_cut.append(False)

        if cut_level not in all_cut_positions[sid]:
            continue

        cut_position = all_cut_positions[sid][cut_level]
        cut_deltas = session_deltas[sid][:cut_position]
        if len(cut_deltas) < 2:
            continue

        G_cut = session_to_nx_original(sid, {sid: cut_deltas}, label_to_int)
        if G_cut.number_of_edges() == 0:
            continue

        graphs_augmented.append(G_cut)
        labels_augmented.append(labels[sid])
        source_ids.append(sid)
        is_cut.append(True)

    print(f"Total graphs after augmentation: {len(graphs_augmented)}")
    print(f"Complete sessions: {sum(1 for c in is_cut if not c)}")
    print(f"Cut sessions: {sum(1 for c in is_cut if c)}")
    print(f"Fraud graphs: {sum(1 for l in labels_augmented if l == 'fraud')}")
    print(f"Normal graphs: {sum(1 for l in labels_augmented if l == 'normal')}")

    return graphs_augmented, labels_augmented, source_ids, is_cut


def compute_fraud_lift_map(cut_level, training_session_ids, training_label, session_deltas, all_cut_positions):
    """Compute fraud lift per transition: fraction_in_fraud - fraction_in_normal, in ~[-1, 1]."""
    fraud_transitions_per_session = []
    normal_transitions_per_session = []

    for sid, label in zip(training_session_ids, training_label):
        cut_position = all_cut_positions[sid].get(cut_level)
        raw = session_deltas[sid]
        if cut_position is not None:
            raw = raw[:cut_position]

        transitions_in_session = set()
        for i in range(1, len(raw)):
            transitions_in_session.add((raw[i - 1][0], raw[i][0]))

        if label == 'fraud':
            fraud_transitions_per_session.append(transitions_in_session)
        else:
            normal_transitions_per_session.append(transitions_in_session)

    n_fraud = len(fraud_transitions_per_session)
    n_normal = len(normal_transitions_per_session)
    if n_fraud == 0 or n_normal == 0:
        return {}

    all_transitions = set()
    for s in fraud_transitions_per_session:
        all_transitions.update(s)
    for s in normal_transitions_per_session:
        all_transitions.update(s)

    fraud_lift_map = {}
    for transition in all_transitions:
        fraud_count = sum(1 for s in fraud_transitions_per_session if transition in s)
        normal_count = sum(1 for s in normal_transitions_per_session if transition in s)
        fraud_lift_map[transition] = fraud_count / n_fraud - normal_count / n_normal

    return fraud_lift_map


def compute_node_features(graph, sequence, sid, active_feature_columns, logs_grouped, imputation_values):
    """Compute per-node scalar features for a per-log graph.

    Returns empty dict when active_feature_columns is empty; the model then runs
    on event embeddings + edge features only.
    """
    if len(active_feature_columns) == 0:
        return {}

    cumulative_times = []
    std_dts_so_far = []
    unique_events_so_far = []
    event_frequency_rank = []
    repeat_indicator = []
    recent_vs_global_dt_ratio = []
    recent_new_event_rate = []
    dt_zscore_so_far = []
    idle_count_so_far = []
    fraction_fast_transitions = []
    event_entropy_so_far = []
    events_since_last_new_event = []

    idle_pause_seconds = 30
    fast_transition_seconds = 0.5
    recent_window_size = 5

    if "cumulative_time" in active_feature_columns:
        running_sum = 0
        for _, dt in sequence:
            running_sum += float(dt)
            cumulative_times.append(running_sum)

    if "std_dt_so_far" in active_feature_columns:
        deltas_seen = []
        for _, dt in sequence:
            deltas_seen.append(float(dt))
            std_dts_so_far.append(0 if len(deltas_seen) < 2 else float(np.std(deltas_seen)))

    if "unique_events_so_far" in active_feature_columns:
        events_seen = set()
        for event_name, _ in sequence:
            events_seen.add(event_name)
            unique_events_so_far.append(len(events_seen))

    if "event_frequency_rank" in active_feature_columns:
        event_counts = {}
        for event_name, _ in sequence:
            event_counts[event_name] = event_counts.get(event_name, 0) + 1
            sorted_events = sorted(event_counts.items(), key=lambda x: (-x[1], x[0]))
            current_event_rank = next(rank for rank, (name, _) in enumerate(sorted_events, start=1) if name == event_name)
            event_frequency_rank.append(float(current_event_rank))

    if "repeat_indicator" in active_feature_columns:
        seen_events = set()
        for event_name, _ in sequence:
            repeat_indicator.append(1 if event_name in seen_events else 0)
            seen_events.add(event_name)

    if "recent_vs_global_dt_ratio" in active_feature_columns:
        deltas = [float(dt) for _, dt in sequence]
        for i in range(len(sequence)):
            recent_window = deltas[max(0, i - recent_window_size + 1):i + 1]
            global_window = deltas[:i+1]
            mean_recent = float(np.mean(recent_window)) if recent_window else 0
            mean_global = float(np.mean(global_window)) if global_window else 1
            recent_vs_global_dt_ratio.append(mean_recent / max(mean_global, 1e-6))

    if "recent_new_event_rate" in active_feature_columns:
        events_seen_so_far = set()
        was_first_time = []
        for event_name, _ in sequence:
            was_first_time.append(event_name not in events_seen_so_far)
            events_seen_so_far.add(event_name)
        for i in range(len(sequence)):
            recent_window = was_first_time[max(0, i - recent_window_size + 1):i+1]
            recent_new_event_rate.append(sum(recent_window) / len(recent_window) if recent_window else 0)

    if "dt_zscore_so_far" in active_feature_columns:
        deltas = [float(dt) for _, dt in sequence]
        for i in range(len(sequence)):
            history = deltas[:i+1]
            if len(history) < 2:
                dt_zscore_so_far.append(0)
            else:
                mean_h, std_h = float(np.mean(history)), float(np.std(history))
                dt_zscore_so_far.append(0 if std_h < 1e-6 else (deltas[i] - mean_h) / std_h)

    if "idle_count_so_far" in active_feature_columns:
        count = 0
        for _, dt in sequence:
            if float(dt) > idle_pause_seconds:
                count += 1
            idle_count_so_far.append(count)

    if "fraction_fast_transitions" in active_feature_columns:
        fast_count = 0
        for i, (_, dt) in enumerate(sequence):
            if float(dt) < fast_transition_seconds:
                fast_count += 1
            fraction_fast_transitions.append(fast_count / (i + 1))

    if "event_entropy_so_far" in active_feature_columns:
        event_counts = {}
        for event_name, _ in sequence:
            event_counts[event_name] = event_counts.get(event_name, 0) + 1
            total = sum(event_counts.values())
            entropy = -sum((c / total) * math.log1p(c / total) for c in event_counts.values())
            event_entropy_so_far.append(entropy)

    if "events_since_last_new_event" in active_feature_columns:
        seen = set()
        steps_since_new = 0
        for event_name, _ in sequence:
            if event_name not in seen:
                seen.add(event_name)
                steps_since_new = 0
            else:
                steps_since_new += 1
            events_since_last_new_event.append(float(steps_since_new))

    synthetic_features = {
        "cumulative_time", "std_dt_so_far", "unique_events_so_far",
        "event_frequency_rank", "repeat_indicator",
        "recent_vs_global_dt_ratio", "recent_new_event_rate", "dt_zscore_so_far",
        "idle_count_so_far", "fraction_fast_transitions", "event_entropy_so_far",
        "events_since_last_new_event",
    }
    df_features = [col for col in active_feature_columns if col not in synthetic_features]

    session_rows = None
    if df_features and logs_grouped is not None:
        session_rows = logs_grouped.get_group(sid).reset_index(drop=True)

    features = {}
    for node in graph.nodes():
        per_node_features = {}
        for col in active_feature_columns:
            if col == 'cumulative_time':
                per_node_features[col] = math.log1p(cumulative_times[node]) if node < len(cumulative_times) else 0
            elif col == 'std_dt_so_far':
                per_node_features[col] = math.log1p(std_dts_so_far[node]) if node < len(std_dts_so_far) else 0
            elif col == 'unique_events_so_far':
                per_node_features[col] = math.log1p(unique_events_so_far[node]) if node < len(unique_events_so_far) else 0
            elif col == "event_frequency_rank":
                per_node_features[col] = event_frequency_rank[node] if node < len(event_frequency_rank) else 0
            elif col == 'repeat_indicator':
                per_node_features[col] = repeat_indicator[node] if node < len(repeat_indicator) else 0
            elif col == "recent_vs_global_dt_ratio":
                per_node_features[col] = math.log1p(recent_vs_global_dt_ratio[node]) if node < len(recent_vs_global_dt_ratio) else 0
            elif col == "recent_new_event_rate":
                per_node_features[col] = recent_new_event_rate[node] if node < len(recent_new_event_rate) else 0
            elif col == "dt_zscore_so_far":
                per_node_features[col] = dt_zscore_so_far[node] if node < len(dt_zscore_so_far) else 0
            elif col == "idle_count_so_far":
                per_node_features[col] = math.log1p(idle_count_so_far[node]) if node < len(idle_count_so_far) else 0
            elif col == "fraction_fast_transitions":
                per_node_features[col] = fraction_fast_transitions[node] if node < len(fraction_fast_transitions) else 0
            elif col == "event_entropy_so_far":
                per_node_features[col] = event_entropy_so_far[node] if node < len(event_entropy_so_far) else 0
            elif col == "events_since_last_new_event":
                per_node_features[col] = math.log1p(events_since_last_new_event[node]) if node < len(events_since_last_new_event) else 0
            else:
                if session_rows is None or node >= len(session_rows):
                    per_node_features[col] = imputation_values.get(col, 0)
                    continue
                value = session_rows.iloc[node][col]
                per_node_features[col] = imputation_values.get(col, 0) if pd.isna(value) else float(value)
        features[node] = per_node_features

    return features


def build_session_graph(sid, session_deltas, cut_position=None, reverse_edges: bool = False):
    """Build a per-log-occurrence graph. One node per log entry, edges carry time deltas."""
    raw = session_deltas[sid]
    if cut_position is not None:
        raw = raw[:cut_position]

    G = nx.DiGraph()
    for i in range(len(raw) - 1):
        G.add_node(i, log_index=i)

    for i in range(len(raw) - 1):
        delta = float(raw[i + 1][1])
        G.add_edge(i, i + 1, mean_dt=delta)
        if reverse_edges:
            G.add_edge(i + 1, i, mean_dt=delta)
    return G, raw


def session_to_pyg_data(sid, session_deltas, cut_position, label, label_to_int, active_feature_columns,
                        logs_grouped=None, imputation_values=None, fraud_lift_map=None, reverse_edges: bool = False):
    """Convert a session into a torch_geometric.data.Data object."""
    G, raw = build_session_graph(sid, session_deltas, cut_position, reverse_edges)
    if G.number_of_edges() == 0:
        return None

    nodes = sorted(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    event_ids = [label_to_int.get(raw[n][0], 0) for n in nodes]

    feat_dict = compute_node_features(G, raw, sid, active_feature_columns, logs_grouped, imputation_values)
    if feat_dict:
        feature_names = list(next(iter(feat_dict.values())).keys())
        scalar_feats = np.array([[feat_dict[n][k] for k in feature_names] for n in nodes], dtype=np.float32)
    else:
        scalar_feats = np.zeros((len(nodes), 0), dtype=np.float32)

    edge_index_list = []
    edge_feats_list = []
    for u, v, d in G.edges(data=True):
        edge_index_list.append([node_idx[u], node_idx[v]])
        log_dt = math.log1p(d.get("mean_dt", 0))
        if fraud_lift_map is not None:
            fraud_lift = fraud_lift_map.get((raw[u][0], raw[v][0]), 0)
            edge_feats_list.append([log_dt, fraud_lift])
        else:
            edge_feats_list.append([log_dt])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_feats_list, dtype=torch.float32)

    return Data(
        event_ids=torch.tensor(event_ids, dtype=torch.long),
        x_scalar=torch.tensor(scalar_feats, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.float32),
        sid=sid,
        num_nodes=len(nodes),
    )


def build_fold_data(indices, cut_level, sids, labels, is_cut_list, session_deltas, all_cut_positions,
                    label_to_int, active_feature_columns, logs_grouped, imputation_values, fraud_lift_map, reverse_edges):
    """Build PyG Data objects for a subset of the training pool."""
    fold_data = []
    for i in indices:
        sid = sids[i]
        lab = labels[i]
        cp = all_cut_positions[sid].get(cut_level) if is_cut_list[i] else None
        d = session_to_pyg_data(sid, session_deltas, cp, 1 if lab == 'fraud' else 0, label_to_int,
                                active_feature_columns, logs_grouped, imputation_values,
                                fraud_lift_map=fraud_lift_map, reverse_edges=reverse_edges)
        if d is not None:
            fold_data.append(d)
    return fold_data


def load_and_prepare(data_path=DATA_PATH):
    df = pd.read_parquet(data_path).sort_values([COL_CASE, COL_TIME]).reset_index(drop=True)
    ses_label = df.groupby(COL_CASE)["label"].first().to_dict()
    session_deltas = compute_time_deltas(df)
    label_to_int = {lab: i for i, lab in enumerate(sorted(set(e for s in session_deltas.values() for e, _ in s)))}
    all_cut_positions = compute_cut_positions(session_deltas, SURROGATE_ACTIVITY)
    _, labels_augmented, source_ids, is_cut = build_augmented_training_graphs(
        list(session_deltas.keys()), session_deltas, all_cut_positions, ses_label, label_to_int, 3, 1)
    return session_deltas, all_cut_positions, label_to_int, source_ids, labels_augmented, is_cut

