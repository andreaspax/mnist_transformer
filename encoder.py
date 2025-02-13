import torch
import attention

class Encoder(torch.nn.Module):
    def __init__(self, input_dim=196, dff=1024, seq_len=16, d_model=64, dropout=0.1):
        super().__init__()

         # Learnable class token
        self.cls_token = torch.nn.parameter.Parameter(torch.randn(1, d_model))

        self.embedding = torch.nn.Linear(input_dim, d_model)
        self.positional_encoding = torch.nn.parameter.Parameter(torch.randn(seq_len+1, d_model))

        self.encoder_block = torch.nn.Sequential(
            EncoderBlock(dff, d_model, dropout),
            EncoderBlock(dff, d_model, dropout),
            EncoderBlock(dff, d_model, dropout),
        )

    def forward(self, x):
        batch_size = x.shape[0]


        x = self.embedding(x)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
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

        # feed-forward with residual connection
        return x + self.linear(x)
    

class Classification(torch.nn.Module):
    def __init__(self, d_model=64, dff=1024, seq_len=16, input_dim=196, output_dim=10):
        super().__init__()
        self.encoder = Encoder(input_dim, dff, seq_len, d_model)
        self.norm = torch.nn.LayerNorm(d_model)
        self.classifier = torch.nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.classifier(x)
        