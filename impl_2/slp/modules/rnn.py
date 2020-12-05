import torch.nn as nn
import torch

from slp.modules.attention import Attention
from slp.modules.embed import Embed
from slp.modules.helpers import PackSequence, PadPackedSequence

from slp.modules.util import pad_mask


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True,
                 layers=1, bidirectional=False, merge_bi='cat', dropout=0,
                 rnn_type='lstm', packed_sequence=True, device='cpu'):

        super(RNN, self).__init__()
        self.device = device
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.merge_bi = merge_bi
        self.rnn_type = rnn_type.lower()

        rnn_cls = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(input_size,
                           hidden_size,
                           batch_first=batch_first,
                           num_layers=layers,
                           bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.packed_sequence = packed_sequence
        if packed_sequence:
            self.pack = PackSequence(batch_first=batch_first)
            self.unpack = PadPackedSequence(batch_first=batch_first)

    def _merge_bi(self, forward, backward):
        if self.merge_bi == 'sum':
            return forward + backward
        return torch.cat((forward, backward), dim=-1)

    def _select_last_unpadded(self, out, lengths):
        gather_dim = 1 if self.batch_first else 0
        gather_idx = ((lengths - 1)  # -1 to convert to indices
                      .unsqueeze(1)  # (B) -> (B, 1)
                      .expand((-1, self.hidden_size))  # (B, 1) -> (B, H)
                      # (B, 1, H) if batch_first else (1, B, H)
                      .unsqueeze(gather_dim))
        # Last forward for real length or seq (unpadded tokens)
        last_out = out.gather(gather_dim, gather_idx).squeeze(gather_dim)
        return last_out

    def _final_output(self, out, lengths):
        # Collect last hidden state
        # Code adapted from https://stackoverflow.com/a/50950188
        if not self.bidirectional:
            return self._select_last_unpadded(out, lengths)

        forward, backward = (out[..., :self.hidden_size],
                             out[..., self.hidden_size:])
        # Last backward corresponds to first token
        last_backward_out = (backward[:, 0, :]
                             if self.batch_first
                             else backward[0, ...])
        # Last forward for real length or seq (unpadded tokens)
        last_forward_out = self._select_last_unpadded(forward, lengths)
        return self._merge_bi(last_forward_out, last_backward_out)

    def forward(self, x, lengths):
        self.rnn.flatten_parameters()
        if self.packed_sequence:
            x, lengths = self.pack(x, lengths)
        out, hidden = self.rnn(x)
        if self.packed_sequence:
            out = self.unpack(out, lengths)
        out = self.drop(out)

        last_timestep = self._final_output(out, lengths)
        return out, last_timestep, hidden


class WordRNN(nn.Module):
    def __init__(
            self, hidden_size, embeddings,
            embeddings_dropout=.1, finetune_embeddings=False,
            batch_first=True, layers=1, bidirectional=False, merge_bi='cat',
            dropout=0.1, rnn_type='lstm', packed_sequence=True,
            attention=False, device='cpu'):
        super(WordRNN, self).__init__()
        self.device = device
        self.embed = Embed(embeddings.shape[0],
                           embeddings.shape[1],
                           embeddings=embeddings,
                           dropout=embeddings_dropout,
                           trainable=finetune_embeddings)
        self.rnn = RNN(
            embeddings.shape[1], hidden_size,
            batch_first=batch_first, layers=layers, merge_bi=merge_bi,
            bidirectional=bidirectional, dropout=dropout,
            rnn_type=rnn_type, packed_sequence=packed_sequence)
        self.out_size = hidden_size if not bidirectional else 2 * hidden_size
        self.attention = None
        if attention:
            self.attention = Attention(
                attention_size=self.out_size, dropout=dropout)

    def forward(self, x, lengths):
        x = self.embed(x)
        out, last_hidden, _ = self.rnn(x, lengths)
        if self.attention is not None:
            out, _ = self.attention(
                out, attention_mask=pad_mask(lengths, device=self.device))
            out = out.sum(1)
        else:
            out = last_hidden
        return out


class Encoder(nn.Module):

    def __init__(self, hidden_size, embeddings, embeddings_dropout=.1,
                 finetune_embeddings=False,
            batch_first=True, layers=1, bidirectional=False, merge_bi='cat',
            dropout=0.1, rnn_type='lstm', packed_sequence=True,
            attention=False, device='cpu'):

        super(Encoder, self).__init__()
        self.device = device
        self.encoder = WordRNN(hidden_size, embeddings, embeddings_dropout=.1,
                 finetune_embeddings=False,
            batch_first=True, layers=1, bidirectional=False, merge_bi='cat',
            dropout=0.1, rnn_type='lstm', packed_sequence=True,
            attention=False, device='cpu')


    def forward(self, input_seq, input_lengths, hidden=None):
            enc_out = self.encoder(input_seq,input_lengths)
            return enc_out


class Decoder(nn.Module):
    def __init__(self, output_size, max_target_len, hidden_size, embeddings,
                 embeddings_dropout=.1,
                 finetune_embeddings=False,
            batch_first=True, layers=1, bidirectional=False, merge_bi='cat',
            dropout=0.1, rnn_type='lstm', packed_sequence=True,
            attention=False, device='cpu'):

        super(Decoder, self).__init__()
        self.device = device
        self.decoder = WordRNN(hidden_size, embeddings, embeddings_dropout=.1,
                 finetune_embeddings=False,
            batch_first=True, layers=1, bidirectional=False, merge_bi='cat',
            dropout=0.1, rnn_type='lstm', packed_sequence=True,
            attention=False, device='cpu')
        self.max_target_len = max_target_len
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, dec_input, dec_hidden):

        dec_out = self.decoder(dec_input, dec_hidden)
        # decoder output equals to decoder hidden
        out = self.linear(dec_out)
        # we return the output and the decoder hidden state
        return out, dec_out


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder,
                 teacher_forcing_ratio=0, device='cpu'):
        super(EncoderDecoder, self).__init__()

        # initialize the encoder and decoder
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.max_target_len = self.decoder.max_target_len
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, input_seq, lengths_inputs, target_seq):

        batch_size = input_seq.shape[0]

        encoder_output = self.encoder(input_seq, lengths_inputs)
        print(encoder_output)

        # decoder_input = [[self.vocloader.word2idx['<SOT>'] for _ in range(
        #     batch_size)]]

        """
        decoder_input = torch.LongTensor(decoder_input)

        decoder_input = decoder_input.transpose(0, 1)
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]
        decoder_input = decoder_input.cuda()
        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() \
                                      < self.teacher_forcing_ratio else False

        decoder_all_outputs = []
        if use_teacher_forcing:

            for t in range(0, self.max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input,
                    decoder_hidden)

                decoder_all_outputs.append(
                    torch.squeeze(decoder_output, dim=1))
                # Teacher forcing: next input is current target
                decoder_input = target_seq[:, t]
                decoder_input = torch.unsqueeze(decoder_input, dim=1)

        else:
            for t in range(0, self.max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input,
                    decoder_hidden)

                decoder_all_outputs.append(
                    torch.squeeze(decoder_output, dim=1))

                # No Teacher forcing: next input is previous output
                current_output = torch.squeeze(decoder_output, dim=1)
                # print("current out",current_output.shape)
                # _, topi = current_output.topk(1)
                # print("topi: ",topi.shape)
                # topi = torch.squeeze(topi, dim=1)

                top_index = F.softmax(current_output, dim=1)
                value, pos_index = top_index.max(dim=1)
                # top1 = current_output.topk(1)
                # print("top1     ",top1)
                # print("pos_index    ",pos_index)
                decoder_input = [index for index in pos_index]
                decoder_input = torch.LongTensor(decoder_input)
                decoder_input = torch.unsqueeze(decoder_input, dim=1)
                decoder_input = decoder_input.cuda()

        return decoder_all_outputs, decoder_hidden

    def evaluate(self, input_seq, lengths_inputs):

        batch_size = input_seq.shape[0]

        encoder_output, encoder_hidden = self.encoder(input_seq,
                                                      lengths_inputs)

        decoder_input = [[self.vocloader.word2idx['<SOT>'] for _ in range(
            batch_size)]]

        decoder_input = torch.LongTensor(decoder_input)

        decoder_input = decoder_input.transpose(0, 1)
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]
        decoder_input = decoder_input.cuda()

        decoder_all_outputs = []

        for t in range(0, self.max_target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden)

            decoder_all_outputs.append(torch.squeeze(decoder_output,
                                                     dim=1).tolist())

            # No Teacher forcing: next input is previous output
            current_output = torch.squeeze(decoder_output, dim=1)
            # print("current out",current_output.shape)
            # _, topi = current_output.topk(1)
            # print("topi: ",topi.shape)
            # topi = torch.squeeze(topi, dim=1)

            top_index = F.softmax(current_output, dim=0)
            value, pos_index = top_index.max(dim=1)
            # top1 = current_output.topk(1)
            # print("top1     ",top1)
            # print("pos_index    ",pos_index)
            decoder_input = [index for index in pos_index]
            decoder_input = torch.LongTensor(decoder_input)
            decoder_input = torch.unsqueeze(decoder_input, dim=1)
            decoder_input = decoder_input.cuda()

        decoder_all_outputs = torch.FloatTensor(
            decoder_all_outputs).transpose(0, 1)
        return decoder_all_outputs, decoder_hidden
        
        """
