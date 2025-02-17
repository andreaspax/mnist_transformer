import torch
import utils
import attention


class Decoder(torch.nn.Module):
    def __init__(self, d_model=64, dff=1024, vocab_size=12, seq_len=5, dropout=0.1):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.positional_encoding = torch.nn.parameter.Parameter(torch.randn(seq_len, d_model))
        
        self.decoder_blocks = torch.nn.ModuleList([
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
            DecoderBlock(dff, d_model, dropout),
        ])
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        seq_len = x.size(1)
        x = self.embedding(x)
        # Only use positional encoding up to current sequence length
        x = x + self.positional_encoding[:seq_len, :]
        
        for block in self.decoder_blocks:
            x = block(x, y)
            
        return x
    
class DecoderBlock(torch.nn.Module):
    def __init__(self, dff=1024, d_model=64, dropout=0.1):
        super().__init__()

        self.attention = attention.SelfAttention(
            d_model, 
            dropout=dropout, 
            heads=4, 
            causal=True
        )

        self.cross_attention = attention.CrossAttention(
            d_model, 
            dropout=dropout, 
            heads=4
        )
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(d_model, dff),
            torch.nn.ReLU(),
            torch.nn.Linear(dff, d_model),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(d_model),
        )


    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.attention(x)
        x = self.cross_attention(x, y)
        x = x + self.linear(x)
        return x
    
    