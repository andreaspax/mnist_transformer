import torch


class SelfAttention(torch.nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4):
        super().__init__()

        self.d_k = d_model // heads
        self.d_v = self.d_k
        self.heads = heads
        self.scale = torch.sqrt(torch.tensor(self.d_k)).item()

        self.Qs = torch.nn.ModuleList([torch.nn.Linear(self.d_k, self.d_k) for _ in range(heads)])
        self.Ks = torch.nn.ModuleList([torch.nn.Linear(self.d_k, self.d_k) for _ in range(heads)])
        self.Vs = torch.nn.ModuleList([torch.nn.Linear(self.d_v, self.d_v) for _ in range(heads)])

        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def generate_causal_mask(seq_len):
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    def forward(self, x: torch.Tensor, masked=False):
        # split x into self.head, equal sized chunks across the last dimension.
        # dim: batch_size, seq_len, d_model -> list of batch_size, seq_len, d_k
        head_chunks = torch.chunk(x, self.heads, dim=-1)    

        As = []
        for i in range(self.heads):
            Q = self.Qs[i](head_chunks[i])  # dim: batch_size, seq_len, d_k
            K = self.Ks[i](head_chunks[i])  # dim: batch_size, seq_len, d_k
            V = self.Vs[i](head_chunks[i])  # dim: batch_size, seq_len, d_v

            # dim: batch_size, seq_len, seq_len
            A = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            A = torch.softmax(A, dim=-1)
            if masked == True:
                self.mask = self.generate_causal_mask(self.d_k).to(x.device)  # [seq_len, seq_len]
                A = A.masked_fill(self.mask, float('-inf'))
            A = torch.matmul(A, V)  # dim: batch_size, seq_len, d_v
            
            As.append(A)

        A = torch.cat(As, dim=-1)
        A = self.dropout(A)
        # A = A + x  # residual connection
        A = self.layer_norm(A)

        return A
