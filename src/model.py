import torch
from torch import nn
from datetime import datetime

class TextEncoder(nn.Module):
    def __init__(self, in_dim = 300, hidden_size = 200, num_layers = 2):
        super(TextEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True, dropout=0.)

    def forward(self, samples):
        return self.encoder(samples)

class SummaryDecoder(nn.Module):
    def __init__(self, in_dim = 300, hidden_size = 200, num_layers = 2):
        super(SummaryDecoder, self).__init__()
        self.decoder = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True, dropout=0.)

    def forward(self, samples, hidden_state):
        return self.decoder(samples, hidden_state)

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

        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads = self.num_heads, dropout=0.)

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

        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads = self.num_heads, dropout=0.5)

        self.prediction_layers = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_size * 2),
            # nn.Dropout(0.1),
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.vocab_size)
        )

        for layer in self.prediction_layers:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight.data)

    def process_with_beam_search(self, idxes_text, idxtoword, beam_size = 5, alpha = 0.8, rank = 0):
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        print(" ".join(idxtoword[i.item()] for i in  idxes_text[0]))
        text_emb_vec = self.embeddings.get_embeddings(idxes_text)
        encoder_output, (h, c) = self.encoder(text_emb_vec)  # 1 x seq_len_d x hid*2

        start_idx = self.embeddings.start_idx
        end_idx = self.embeddings.end_idx
        inactivated_beams = []
        beam = [[(torch.tensor([start_idx]).to(device), (h,c), 0, 1)] for i in range(beam_size)] # word_idx, hidden_state, log_prob, seq_len
        seq_len = 1
        while len(inactivated_beams) < beam_size and seq_len < 100:
            seq_len += 1
            top_samples = [] #(beam_num, word_idx, hidden_state, log_prob, seq_len)
            for beam_num in range(beam_size):
                last_elem = beam[beam_num][-1]
                if beam[beam_num][-1][0] == end_idx: # If a beam is inactivated
                    top_samples.append((beam_num, *last_elem))
                    continue
                last_word_emb = self.embeddings.get_embeddings(last_elem[0].unsqueeze(0)) # 1 x 1 x 300
                decoder_output, (h,c) = self.decoder(last_word_emb, last_elem[1])

                decoder_attention, attention_weights = self.multihead_attn(decoder_output.transpose(0, 1),
                                                                           encoder_output.transpose(0, 1),
                                                                           encoder_output.transpose(0, 1))
                decoder_attention = decoder_attention.transpose(0, 1)

                pred_vector = torch.cat((decoder_output, decoder_attention), dim=-1)

                pred = self.prediction_layers(pred_vector.reshape(-1, self.hidden_size * 2)) # 1 x C
                pred = torch.softmax(pred, dim=-1)

                vals, indices = torch.topk(pred.view(-1), beam_size)
                for j in range(beam_size):
                    top_samples.append((beam_num, indices[j].unsqueeze(0), (h,c), torch.log(vals[j]) + last_elem[-2], last_elem[-1] + 1))

            # Choose top beam_size samples
            chooser = torch.zeros(0).to(device)
            for sample in top_samples:
                chooser = torch.cat((chooser, torch.tensor([sample[-2] / (sample[-1] ** alpha)]).to(device)), dim=0) # Beam probs with lenght normalizations
            vals, indices = chooser.topk(beam_size)
            for idx in indices:
                sample = top_samples[idx]
                if beam[sample[0]][-1][0] == end_idx: # Already reached end
                    continue
                if sample[1] == end_idx:
                    inactivated_beams.append(sample[0])
                print(idxtoword[sample[1].item()] + " ")
                beam[sample[0]].append(tuple(sample[1:]))

        # Create top beam_size summaries
        summaries = []
        for beam_num in range(beam_size):
            summaries.append(" ".join([idxtoword[itm[0][0].item()] for itm in beam[beam_num][:-1]]))
        return summaries

    def get_attention_output(self, idxes_text, idxes_summary):
        text_emb_vec = self.embeddings.get_embeddings(idxes_text)
        encoder_output, (h, c) = self.encoder(text_emb_vec)  # batch x seq_len_d x hid*2

        summ_emb_vec = self.embeddings.get_embeddings(idxes_summary)
        decoder_output, (h, c) = self.decoder(summ_emb_vec, (h, c))

        decoder_attention, attention_weights = self.multihead_attn(decoder_output.transpose(0, 1),
                                                                   encoder_output.transpose(0, 1),
                                                                   encoder_output.transpose(0,
                                                                                            1))  # batch x seq_len_d x hid * 2
        decoder_attention = decoder_attention.transpose(0, 1) # Batch first
        return decoder_attention

    def forward(self, idxes_text, idxes_summary): # Batch x seq_len_e
        text_emb_vec = self.embeddings.get_embeddings(idxes_text)
        encoder_output, (h,c) = self.encoder(text_emb_vec) # batch x seq_len_d x hid*2

        summ_emb_vec = self.embeddings.get_embeddings(idxes_summary)
        decoder_output, (h,c) = self.decoder(summ_emb_vec, (h,c))

        decoder_attention, attention_weights = self.multihead_attn(decoder_output.transpose(0,1), encoder_output.transpose(0,1), encoder_output.transpose(0,1)) # batch x seq_len_d x hid * 2
        decoder_attention = decoder_attention.transpose(0,1)

        pred_vector = torch.cat((decoder_output, decoder_attention), dim=-1)

        pred = self.prediction_layers(pred_vector.reshape(-1, self.hidden_size * 2))
        return pred.view(idxes_summary.shape[0], idxes_summary.shape[1], -1)


