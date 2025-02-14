import torch


class SelfAttention(torch.nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4, causal=False):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.scale = torch.sqrt(torch.tensor(self.d_k)).item()

        self.keys = torch.nn.Linear(d_model, 3 * d_model)

        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(d_model, d_model)
        self.causal = causal

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # Split into three equal chunks of size d_model each
        qry, key, val = self.keys(x).chunk(3, dim=-1)

        # dim: batch_size, seq_len, d_model -> batch_size, seq_len, heads, d_k
        qry = qry.reshape(batch_size, seq_len, self.heads, self.d_k)
        key = key.reshape(batch_size, seq_len, self.heads, self.d_k)
        val = val.reshape(batch_size, seq_len, self.heads, self.d_k)

        # dim: batch_size, seq_len, heads, d_k -> batch_size, heads, seq_len, d_k
        qry = qry.transpose(1, 2)
        key = key.transpose(1, 2)
        val = val.transpose(1, 2)

        A = torch.matmul(qry, key.transpose(-2, -1)) / self.scale


        if self.causal:
                mask = torch.tril(torch.ones(A.shape[-2:], device=A.device))
                mask = mask.unsqueeze(0)  # add batch dimension
                A = A.masked_fill(mask == 0, float("-inf"))
            
        A = torch.softmax(A, dim=-1)
        A = torch.matmul(A, val)  # dim: batch_size, heads, seq_len, d_k

        A = A.transpose(1, 2)  # dim: batch_size, seq_len, heads, d_k
        A = A.reshape(batch_size, seq_len, self.d_model)

        A = self.out(A)
        A = self.dropout(A)
        A = A + x  # residual connection
        A = self.layer_norm(A)

        return A

class CrossAttention(torch.nn.Module):
    def __init__(self, d_model=24, dropout=0.1, heads=4):
        super().__init__()

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads


        self.qry = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.val = torch.nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.tensor(self.d_k)).item()

        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x is input from decoder, y is input from encoder
        # dim of x, y: batch_size, seq_len, d_model
        batch_size, seq_len_x, _ = x.shape  # 5x64
        batch_size, seq_len_y, _ = y.shape  # 17x64

        qry = self.qry(x)
        key = self.key(y)
        val = self.val(y)

        # reshape qry, key, val to batch_size, seq_len, heads, d_k
        qry = qry.reshape(batch_size, seq_len_x, self.heads, self.d_k)
        key = key.reshape(batch_size, seq_len_y, self.heads, self.d_k)
        val = val.reshape(batch_size, seq_len_y, self.heads, self.d_k)

        # dim: batch_size, seq_len_, heads, d_k -> batch_size, heads, seq_len_, d_k
        qry = qry.transpose(1, 2)
        key = key.transpose(1, 2)
        val = val.transpose(1, 2)

        # compute attention. dim: batch_size, heads, seq_len_x, seq_len_y
        A = torch.matmul(qry, key.transpose(-2, -1)) / self.scale
        A = torch.softmax(A, dim=-1)
        A = torch.matmul(A, val)

        # dim: batch_size, heads, seq_len_x, d_k -> batch_size, seq_len_x, heads, d_k
        A = A.transpose(1, 2)

        # dim: batch_size, seq_len, heads, d_k -> batch_size, seq_len_x, d_model
        A = A.reshape(batch_size, seq_len_x, self.d_model)

        A = self.out(A)
        A = self.dropout(A)
        A = A + x  # residual connection
        A = self.layer_norm(A)

        return A