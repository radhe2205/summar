from torch import nn
from datetime import datetime

class TextEncoder(nn.Module):
    def __init__(self, in_dim = 300, hidden_size = 200, num_layers = 2):
        super(TextEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=0.4)

    def forward(self, samples):
        return self.encoder(samples)

class SummaryDecoder(nn.Module):
    def __init__(self, in_dim = 300, hidden_size = 200, num_layers = 2):
        super(SummaryDecoder, self).__init__()
        self.decoder = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=0.4)

    def forward(self, samples):
        return self.decoder(samples)

class AttentionModel(nn.Module):
    def __init__(self, embeddings, vocab_size = 400004, embedding_dim = 300, hidden_size = 256):
        super(AttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_heads = 4
        self.vocab_size = vocab_size

        self.embeddings = embeddings
        self.encoder = TextEncoder(hidden_size = self.hidden_size)
        self.decoder = SummaryDecoder(hidden_size = self.hidden_size)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.hidden_size * 2, num_heads = self.num_heads, dropout=0.5)

        self.prediction_layers = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_size * 2),
            nn.Dropout(0.1),
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.vocab_size)
        )

        for layer in self.prediction_layers:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight.data)

    def forward(self, idxes_text, idxes_summary): # Batch x seq_len_e
        text_emb_vec = self.embeddings.get_embeddings(idxes_text)
        encoder_output, (h,c) = self.encoder(text_emb_vec) # batch x seq_len_d x hid*2

        summ_emb_vec = self.embeddings.get_embeddings(idxes_summary)
        decoder_output, (h,c) = self.decoder(summ_emb_vec)

        decoder_attention, attention_weights = self.multihead_attn(decoder_output.transpose(0,1), encoder_output.transpose(0,1), encoder_output.transpose(0,1)) # batch x seq_len_d x hid * 2
        decoder_attention = decoder_attention.transpose(0,1)

        pred = self.prediction_layers(decoder_attention.reshape(-1, self.hidden_size * 2))
        return pred.view(idxes_summary.shape[0], idxes_summary.shape[1], -1)


class AttentionModelLimited(nn.Module):
    def __init__(self, embeddings, vocab_size = 400004, embedding_dim = 300, hidden_size = 256):
        super(AttentionModelLimited, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_heads = 4
        self.vocab_size = vocab_size

        self.embeddings = embeddings
        self.encoder = TextEncoder(hidden_size = self.hidden_size)
        self.decoder = SummaryDecoder(hidden_size = self.hidden_size)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.hidden_size * 2, num_heads = self.num_heads, dropout=0.1)

        self.prediction_layers = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_size * 2),
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size * 3),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size * 3, out_features=self.vocab_size)
        )

        for layer in self.prediction_layers:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight.data)

    def forward(self, idxes_text, idxes_summary): # Batch x seq_len_e
        text_emb_vec = self.embeddings.get_embeddings(idxes_text)
        encoder_output, (h,c) = self.encoder(text_emb_vec) # batch x seq_len_d x hid*2

        summ_emb_vec = self.embeddings.get_embeddings(idxes_summary)
        decoder_output, (h,c) = self.decoder(summ_emb_vec)

        decoder_attention, attention_weights = self.multihead_attn(decoder_output.transpose(0,1), encoder_output.transpose(0,1), encoder_output.transpose(0,1)) # batch x seq_len_d x hid * 2
        decoder_attention = decoder_attention.transpose(0,1)

        pred = self.prediction_layers(decoder_attention.reshape(-1, self.hidden_size * 2))
        return pred.view(idxes_summary.shape[0], idxes_summary.shape[1], -1)


