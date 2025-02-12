import torch
import attention

class Encoder(torch.nn.Module):
    def __init__(self, input_dim=196, dff=1024, seq_len=16, d_model=64, dropout=0.1):
        super().__init__()

        self.embedding = torch.nn.Linear(input_dim, d_model)
        self.positional_encoding = torch.nn.parameter.Parameter(torch.randn(seq_len, d_model))

        self.encoder_block = torch.nn.Sequential(
            EncoderBlock(dff, d_model, dropout),
            # EncoderBlock(dff, d_model, dropout),
            # EncoderBlock(dff, d_model, dropout),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding

        x = self.encoder_block(x)

        return x
        
class EncoderBlock(torch.nn.Module):
    def __init__(self, dff=256, d_model=64, dropout=0.1):
        super().__init__()

        self.attention = attention.SelfAttention(d_model, dropout)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(d_model, dff),
            torch.nn.ReLU(),
            torch.nn.Linear(dff, d_model),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # multi-head attention with residual connection
        x = self.attention(x)

        x = self.linear(x)

        # feed-forward with residual connection
        return x
        
        