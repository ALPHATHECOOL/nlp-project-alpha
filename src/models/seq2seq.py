# In seq2seq.py (for example):

import torch
import torch.nn as nn
from .lstm_cells import VanillaLSTMCell, PeepholeLSTMCell, WorkingMemoryLSTMCell


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, lstm_cell_type='vanilla', dropout_rate=0.3):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout_rate)

        if lstm_cell_type == 'vanilla':
            self.lstm_cell = VanillaLSTMCell(embed_size, hidden_size)
        elif lstm_cell_type == 'peephole':
            self.lstm_cell = PeepholeLSTMCell(embed_size, hidden_size)
        elif lstm_cell_type == 'working_memory':
            self.lstm_cell = WorkingMemoryLSTMCell(embed_size, hidden_size)
        else:
            raise ValueError("Invalid LSTM cell type")

    def forward(self, src):
        batch_size, seq_len = src.size()
        device = src.device
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)

        for t in range(seq_len):
            x_t = self.embedding(src[:, t])
            x_t = self.dropout(x_t)  # Apply dropout to embedding
            h, c = self.lstm_cell(x_t, h, c)
        return h, c


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, lstm_cell_type='vanilla', dropout_rate=0.3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout_rate)

        if lstm_cell_type == 'vanilla':
            self.lstm_cell = VanillaLSTMCell(embed_size, hidden_size)
        elif lstm_cell_type == 'peephole':
            self.lstm_cell = PeepholeLSTMCell(embed_size, hidden_size)
        elif lstm_cell_type == 'working_memory':
            self.lstm_cell = WorkingMemoryLSTMCell(embed_size, hidden_size)
        else:
            raise ValueError("Invalid LSTM cell type")

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, h, c):
        batch_size, seq_len = tgt.size()
        outputs = []
        for t in range(seq_len):
            x_t = self.embedding(tgt[:, t])
            x_t = self.dropout(x_t)  # Apply dropout to embedding
            h, c = self.lstm_cell(x_t, h, c)
            logits = self.fc(h)
            outputs.append(logits.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size, lstm_cell_type='vanilla',
                 dropout_rate=0.3):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size, lstm_cell_type, dropout_rate)
        self.decoder = Decoder(tgt_vocab_size, embed_size, hidden_size, lstm_cell_type, dropout_rate)

    def forward(self, src, tgt):
        h, c = self.encoder(src)
        outputs = self.decoder(tgt, h, c)
        return outputs
