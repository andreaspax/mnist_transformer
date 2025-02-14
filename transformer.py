import torch
import encoder
import decoder

class Transformer(torch.nn.Module):
    def __init__(self, d_model=64, dff=1024, vocab_size=12, seq_len_x=5, seq_len_y=16, dropout=0.1):
        super().__init__()
        
        self.encoder = encoder.Encoder(input_dim=196, dff=dff, seq_len=seq_len_y, d_model=d_model, dropout=dropout)
        self.decoder = decoder.Decoder(d_model=d_model, dff=dff, vocab_size=vocab_size, seq_len=seq_len_x, dropout=dropout)

        self.final = torch.nn.Linear(d_model, vocab_size)

        self.softmax = torch.nn.Softmax(dim=-1)

        
    def forward(self, x, y):
        # Ensure proper data types
        x = x.long()  # For decoder's embedding layer (expects long integers)
        y = y.float()  # For encoder's linear layer (expects floats)
        
        encoder_output = self.encoder(y)
        decoder_output = self.decoder(x, encoder_output)
        decoder_output = self.final(decoder_output)
        decoder_output = self.softmax(decoder_output)
        return decoder_output

class NextTokenPredictor(torch.nn.Module):
    def __init__(self):
        super(NextTokenPredictor, self).__init__()
        self.fc = torch.nn.Linear(12, 12)  # Linear layer for next token prediction

    def forward(self, x):
        x = self.fc(x)
        return x

    