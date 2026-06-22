"""Model definitions: GINE (graph) vs LSTM / Transformer (sequence) baselines.

All three consume the same PyG `Data` batch; the sequence models flatten it back into padded
per-session sequences and ignore the graph structure.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool


class GINEFraudModel(nn.Module):
    """GINE model for fraud detection.

    - Event embedding concatenated with scalar node features → hidden_dim via linear proj
    - Edge features projected to hidden_dim (required by GINEConv)
    - n_layers GINEConv layers with BatchNorm + ReLU + Dropout
    - Mean+max global pooling → 2-layer MLP head → single logit
    """

    def __init__(self, n_event_types, n_scalar_node_features, n_edge_features=2,
                 event_emb_dim=16, hidden_dim=32, n_layers=2, dropout=0.4):
        super().__init__()
        self.event_emb = nn.Embedding(n_event_types, event_emb_dim)
        self.node_proj = nn.Linear(event_emb_dim + n_scalar_node_features, hidden_dim)
        self.edge_proj = nn.Linear(n_edge_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        ev_ids = batch.event_ids.clamp(max=self.event_emb.num_embeddings - 1)
        ev_emb = self.event_emb(ev_ids)

        if batch.x_scalar.shape[-1] > 0:
            x = torch.cat([ev_emb, batch.x_scalar], dim=-1)
        else:
            x = ev_emb
        x = self.node_proj(x)
        e = self.edge_proj(batch.edge_attr)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, batch.edge_index, e)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h_mean = global_mean_pool(x, batch.batch)
        h_max = global_max_pool(x, batch.batch)
        return self.head(torch.cat([h_mean, h_max], dim=-1)).squeeze(-1)


class LSTMFraudModel(nn.Module): 
    """ LSTM baseline
 
        Same input as GINEFRaudModl, but hadnels the data differently, the grraph structure is ignored.
    """
 
    def __init__(self, n_event_types : int, n_scalar_node_features : int, event_emb_dim: int = 16, hidden_dim: int = 64, n_layers : int = 2, dropout : float = 0.3, bidirectional = False): 
                
        super().__init__()#calls the constructor of the parent class (nn.Module)
        
        self.event_emb = nn.Embedding(n_event_types, event_emb_dim)
 
        token_input_dim = event_emb_dim + n_scalar_node_features
        self.lstm = nn.LSTM(token_input_dim, hidden_dim, num_layers  = n_layers, dropout = dropout if n_layers > 1 else 0, batch_first = True, bidirectional = bidirectional)
        head_input_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, batch): 
        # Look up event embeddings, clamp ids to embedding range
        ev_ids = batch.event_ids.clamp(max=self.event_emb.num_embeddings-1) # set a max range and min not set here
        ev_emb = self.event_emb(ev_ids)
 
        if batch.x_scalar.shape[-1] > 0: 
            x = torch.cat([ev_emb, batch.x_scalar], dim =-1)
        else: 
            x = ev_emb
        #Reshape form flat (total_nodes, dim) to padded (batc_size, max_len, dim) using batch.batch to identify which session each node belongs to
        batch_size = int(batch.batch.max().item()) + 1
        sessions = []
        lengths = []
        for session_id in range(batch_size): 
            session_mask = batch.batch == session_id
            session_tokens = x[session_mask]
            sessions.append(session_tokens)
            lengths.append(session_tokens.size(0))
 
        max_len = max(lengths)
        padded = torch.zeros(batch_size, max_len, x.size(-1), device = x.device, dtype = x.dtype)
        for i, session_tokens in enumerate(sessions): 
            padded[i, :lengths[i]] = session_tokens
        
        # pack for LSTM efficiency (skips padding in computation)
        packed = nn.utils.rnn.pack_padded_sequence(padded, lengths, batch_first = True, enforce_sorted = False)
        _, (hidden, _) = self.lstm(packed)
 
        # for bidirectional lstm, concatenate forward and backward final states
        if self.lstm.bidirectional: 
            # hidden shape : (num_layers * 2, batch, hidden_dim)
            # take last layers forward and backward states 
            final = torch.cat([hidden[-2], hidden[-1]], dim = -1)
        else: 
            final = hidden[-1]
        return self.head(final).squeeze(-1)


class TransformerFraudModel(nn.Module): 
    """ Transformer baseline
 
        Same input as GINEFRaudModl, and TransformerFraudModel but hadnels the data differently, the grraph structure is ignored.
        Consumees the same batched data objects as LSTM but reads thedm azs sequences.
        
    """
 
    def __init__(self, n_event_types : int, n_scalar_node_features : int, event_emb_dim: int = 16, hidden_dim: int = 64, n_layers : int = 2, n_heads : int = 4,  dropout : float = 0.3, max_sequence_length = 5000): 
                
        super().__init__()#calls the constructor of the parent class (nn.Module)
        
        self.event_emb = nn.Embedding(n_event_types, event_emb_dim)
 
        token_input_dim = event_emb_dim + n_scalar_node_features
        # project to hidden_dim so transformer dimensions are clean
        self.token_proj = nn.Linear(token_input_dim, hidden_dim)
 
        # sinusoidal positional encoding - fixed not learned (precomputed)
        positional_encoding = self._build_sinusoidal_encoding(max_sequence_length, hidden_dim)
        self.register_buffer('positional_encoding', positional_encoding)
 
        encode_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, dropout = dropout, batch_first = True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encode_layer, num_layers=n_layers)
 
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, 1),
        )
    @staticmethod
    def _build_sinusoidal_encoding(max_length: int, dim: int): 
        """ Standard sinusoidal positional encoding  """
        position =torch.arange(max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0)/ dim))
        pe = torch.zeros(max_length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
 
    def forward(self, batch): 
        # Look up event embeddings, clamp ids to embedding range
        ev_ids = batch.event_ids.clamp(max=self.event_emb.num_embeddings-1) # set a max range and min not set here
        ev_emb = self.event_emb(ev_ids)
 
        if batch.x_scalar.shape[-1] > 0: 
            x = torch.cat([ev_emb, batch.x_scalar], dim =-1)
        else: 
            x = ev_emb
        tokens = self.token_proj(x)
 
        #Reshape form flat (total_nodes, dim) to padded (batc_size, max_len, dim) using batch.batch to identify which session each node belongs to
        batch_size = int(batch.batch.max().item()) + 1
        sessions = []
        lengths = []
        for session_id in range(batch_size): 
            session_mask = batch.batch == session_id
            session_tokens = tokens[session_mask]
            sessions.append(session_tokens)
            lengths.append(session_tokens.size(0))
 
        max_len = max(lengths)
        padded = torch.zeros(batch_size, max_len, tokens.size(-1), device = tokens.device, dtype = tokens.dtype)
        attention_padding_mask = torch.ones(batch_size, max_len, dtype = torch.bool, device = tokens.device, )
 
        for i, session_tokens in enumerate(sessions): 
            session_length = lengths[i]
            padded[i, :session_length] = session_tokens
            attention_padding_mask[i, :session_length] = False # false = attend
        
        # Add positional encoding (truncate to current max_len)
        padded = padded + self.positional_encoding[:max_len].unsqueeze(0)
 
        #  " pass through encoder" mask padding so it's ignored in attention
        encoded = self.encoder(padded, src_key_padding_mask=attention_padding_mask)
 
        # Mean pool over non-padded positions to get one vector per session
        mask = (~attention_padding_mask).unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim = 1) / mask.sum(dim = 1).clamp(min=1)
 
        return self.head(pooled).squeeze(-1)

