from MovieCorpus_Dataloader import MovieCorpusDataloader
from text_preprocessor import TextProcessor
from Text_Tokenizer import TextTokenizer
from Voc import EmbVoc
from IndexesLoader import IndexesLoader
from Padder import DataPadderMasker
from QADataset import QADataset
from BatchLoader import Batchloader
from Seq2Seq import *
from utils import *

import numpy as np

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

x = MovieCorpusDataloader()

quest, ans = x.load_data('/home/manzar/Desktop/diplwmatiki/chatbot/data/'
                         'movie_lines.txt',
                         '/home/manzar/Desktop/diplwmatiki/chatbot/data/'
                         'movie_conversations.txt')

quest = quest[:1000]
ans = ans[:1000]

# concatenate data for easier preprocessing, but keep lengths
quest_len = len(quest)
ans_len = len(ans)
data = quest + ans

# Preprocess data
txtpr = TextProcessor(lowercase=True, cleaning=True)
filt_data = txtpr.process_text(data)

# Split filtered sentences to words
token = TextTokenizer()
data_tok = token.word_tokenization(filt_data)


# Apply some threshold-filtering
min_len= 2
max_len = 25
data_tok,quest_len,ans_len = threshold_filtering(min_len,max_len,data_tok,quest_len)
print(quest_len)
# Create voc with indexes
# at first we create custom embeddings
emb_dim = 300
pad_token_idx = 0
end_token_idx = 2
unk_token_idx = 3

tag_dict = {'PAD': np.zeros(emb_dim), 'SOT': np.random.rand(emb_dim),
            'EOS': np.random.rand(emb_dim), 'UNK': np.random.rand(emb_dim)}

vocloader = EmbVoc(tag_dict)
# vocloader.add_embeddings('/home/manzar/Desktop/diplwmatiki/'
#                          'glove/glove.6B/glove.6B.100d.txt')
vocloader.add_embeddings('/home/manzar/Desktop/diplwmatiki/word2vec'
                         '/GoogleNews-vectors-negative300.bin',sel="gensim")

# load indexes
idxloader = IndexesLoader(vocloader, unk_token=vocloader.word2idx['UNK'],
                          end_token=vocloader.word2idx['EOS'])
indexed_data = idxloader.get_indexes(data_tok)


questions = indexed_data[:quest_len]
answers = indexed_data[quest_len:]

# form data to pairs
pairs = data2pairs(questions, answers)

# now lets pad our data and receive useful info (padded data, lengths of inputs,
# masks of targets, maximum length of target data)
padder = DataPadderMasker(pad_token=vocloader.word2idx['PAD'])

padded_inputs,lengths_inputs,padded_targets, masks_targets,max_len_trg =  \
    padder.pad_data(pairs)


print("max target length: ", max_len_trg)

dataset = QADataset(padded_inputs,lengths_inputs,padded_targets,
                    masks_targets, max_len_trg)

BATCH_SZ_train = 32
BATCH_SZ_val = 3
batchloader = Batchloader()
train_batches, val_batches = batchloader.torch_train_val_split(dataset,
                                                               BATCH_SZ_train,
                                                               BATCH_SZ_val,
                                                               val_size=0.33)


# make the encoder
enc_hidden_size = 100
enc_n_layers = 2
enc = EncoderLSTM(vocloader.embeddings, enc_hidden_size, enc_n_layers,
                  batch_first=True, bidirectional=True)


dec_hidden_size = 100
dec_vocab_size = vocloader.embeddings.shape[0]
dec_output_size = dec_vocab_size
dec_n_layers = 2
max_target_len = max_len_trg


dec = DecoderLSTM(vocloader.embeddings, dec_hidden_size, dec_output_size,
                  max_target_len, dec_n_layers, batch_first=True)






num_epochs = 10
teacher_forcing_rat =0.6
clip=50
criterion = nn.CrossEntropyLoss(ignore_index=vocloader.word2idx['PAD'])

enc_optimizer = torch.optim.Adam(enc.parameters(),lr=0.0001)
dec_optimizer = torch.optim.Adam(dec.parameters(),lr=0.0005)

device=torch.device("cuda")
enc.cuda()
dec.cuda()

for epoch in range(num_epochs):
    # no need to set requires_grad=True for parameters(weights)  as it done
    # by default. Also for input requires_grad is not
    # always necessary. So we comment the following line.
    # with torch.autograd():

    enc.train()
    dec.train()
    # scheduler.step()
    running_average_loss = 0

    # train model in each epoch
    for index, batch in enumerate(train_batches):
        loss = 0
        inputs, lengths_inputs, targets, masks_targets = batch
        inputs = inputs.long().cuda()
        targets = targets.long().cuda()
        lengths_inputs.cuda()
        masks_targets.cuda()

        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        enc_out, enc_hidden = enc(inputs, lengths_inputs)

        # 1.Create initial decoder input (start with SOS tokens for each turn)
        # 2.enc_hidden is a tuple (2nd term is grad)
        # from each layer of the encoder we receive the last hidden state and
        # set it to our decoder hidden state!

        decoder_input = [[vocloader.word2idx['SOT']for _ in range(batch[0].shape[0])]]
        decoder_input = torch.LongTensor(decoder_input)

        decoder_input = decoder_input.transpose(0, 1)
        #print("enc hidden",enc_hidden[0].shape)
        #print("dec input",decoder_input.shape)
        decoder_hidden = enc_hidden[:dec.num_layers]
        decoder_input = decoder_input.cuda()
        dec_outs, dec_hidden = dec(decoder_input, enc_hidden, targets,
                                   teacher_forcing_rat)

        for t in range(0, max_target_len):
            # Calculate and accumulate loss
            #top_index = F.softmax(dec_outs[t], dim=0)
            #value, pos_index = top_index.max(dim=1)

            loss += criterion(dec_outs[t], targets[:,t].long())
        #print("Loss calculated")
        # Perform backpropatation
        loss.backward()
        running_average_loss += loss / max_target_len
        #print("backprop done")


        # Clip gradients: gradients are modified in place
        # _ = nn.utils.clip_grad_norm_(enc.parameters(), clip)
        # _ = nn.utils.clip_grad_norm_(dec.parameters(), clip)

        # Adjust model weights
        enc_optimizer.step()
        dec_optimizer.step()
        last =index
    print("Epoch: {} \t \t Training Loss {}".format(epoch, float(
        running_average_loss) / (last + 1)))

