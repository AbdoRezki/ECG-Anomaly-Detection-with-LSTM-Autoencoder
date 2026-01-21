import torch
import torch.nn as nn

class RecurrentAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_lstm1 = nn.LSTM(1, 128, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.decoder_lstm1 = nn.LSTM(64, 128, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(128, 1, batch_first=True)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Encoder: x [batch,1,140] → [batch,140,1]
        x_enc = x.squeeze(1).unsqueeze(-1)
        _, (hn1, _) = self.encoder_lstm1(x_enc)
        _, (hn2, _) = self.encoder_lstm2(hn1)
        z = hn2[-1]  # [batch,64]
        
        # Decoder
        decoder_input = z.unsqueeze(1).repeat(1, 140, 1)  # [batch,140,64]
        out1, _ = self.decoder_lstm1(decoder_input)
        out2, _ = self.decoder_lstm2(out1)
        recon = out2.squeeze(-1).unsqueeze(1)  # [batch,1,140]
        
        return recon
