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

quest, ans = x.load_data('../data/movie_lines.txt',
                         '../data/movie_conversations.txt')

quest = quest[:400]
ans = ans[:400]

# concatenate data for easier preprocessing, but keep lengths
quest_len = len(quest)
ans_len = len(ans)
data = quest + ans

# Preprocess data
txtpr = TextProcessor(lowercase=True, cleaning=True, remove_punct=True)
filt_data = txtpr.process_text(data)

# Split filtered sentences to words
token = TextTokenizer()
data_tok = token.word_tokenization(filt_data)


# Apply some threshold-filtering
min_len= 2
max_len = 25
data_tok,quest_len,ans_len = threshold_filtering(min_len,max_len,data_tok,
                                                 quest_len)
print(quest_len)
# Create voc with indexes
# at first we create custom embeddings
emb_dim = 100
pad_token_idx = 0
end_token_idx = 2
unk_token_idx = 3

tag_dict = {'<PAD>': np.zeros(emb_dim), '<SOT>': np.random.rand(emb_dim),
            '<EOS>': np.random.rand(emb_dim), '<UNK>': np.random.rand(emb_dim)}

vocloader = EmbVoc(tag_dict)
vocloader.add_embeddings('/home/manzar/Desktop/emp/diplwmatiki/'
                          'glove/glove.6B/glove.6B.100d.txt')
#vocloader.add_embeddings('/home/manzar/Desktop/diplwmatiki/word2vec'
#                         '/GoogleNews-vectors-negative300.bin',sel="gensim")

# load indexes
idxloader = IndexesLoader(vocloader, unk_token=vocloader.word2idx['<UNK>'],
                          end_token=vocloader.word2idx['<EOS>'])
indexed_data = idxloader.get_indexes(data_tok)


questions = indexed_data[:quest_len]
answers = indexed_data[quest_len:]

# form data to pairs
pairs = data2pairs(questions, answers)

# now lets pad our data and receive useful info (padded data, lengths of inputs,
# masks of targets, maximum length of target data)
padder = DataPadderMasker(pad_token=vocloader.word2idx['<PAD>'])

padded_inputs,lengths_inputs,padded_targets, masks_targets,max_len_trg =  \
    padder.pad_data(pairs)


print("max target length: ", max_len_trg)

dataset = QADataset(padded_inputs,lengths_inputs,padded_targets,
                    masks_targets, max_len_trg)

BATCH_SZ_train = 32
BATCH_SZ_val = 1
batchloader = Batchloader()
train_batches, val_batches = batchloader.torch_train_val_split(dataset,
                                                               BATCH_SZ_train,
                                                               BATCH_SZ_val,
                                                               val_size=0.10)


# make the encoder
enc_hidden_size = 100
enc_n_layers = 2
enc = EncoderLSTM(vocloader.embeddings, enc_hidden_size, enc_n_layers,
                  batch_first=True, bidirectional=True)

# make the decoder
dec_hidden_size = 100
dec_vocab_size = vocloader.embeddings.shape[0]
dec_output_size = dec_vocab_size
dec_n_layers = 2
max_target_len = max_len_trg
dec = DecoderLSTM_v2(vocloader.embeddings, dec_hidden_size, dec_output_size,
                     max_target_len, dec_n_layers, batch_first=True)


# make the encoder decoder model
teacher_forcing_rat = 1
model = EncoderDecoder(enc,dec,vocloader,teacher_forcing_rat)
model.cuda()
print(model)


# select optimizers,loss function and clipping value
criterion = nn.CrossEntropyLoss(ignore_index=vocloader.word2idx['<PAD>'],
                                reduction='mean')
enc_optimizer = torch.optim.Adam(enc.parameters(),lr=0.001)
dec_optimizer = torch.optim.Adam(dec.parameters(),lr=0.005)
model_optimizers = [enc_optimizer,dec_optimizer]
clip = 50

num_epochs = 100

# for epoch in range(num_epochs):
#     model.train()
#     avg_epoch_loss = train(train_batches, model, model_optimizers, criterion,
#                            clip)
#     print("Epoch: {} \t \t Training Loss {}".format(epoch, float(
#         avg_epoch_loss)))


model_name = "simple_encdec_lr_0001"
save_dir = "/media/manzar/Data/toshiba_temp/diplwmatiki/chatbot/saved_models"
print_every = 5
save_every = 10
corpus_name = "MovieCorpus_Cornell"
train_epochs(train_batches, model_name, model, model_optimizers, criterion,
             save_dir, num_epochs, print_every, save_every, corpus_name,clip)



# load my model:
# checkpoint = torch.load('/media/manzar/Data/toshiba_temp/diplwmatiki'
#                             '/chatbot'
#                '/saved_models/simple_encdec/MovieCorpus_Cornell/15_checkpoint.tar')
#
#
# model = checkpoint['model']


#validate(val_batches,model)
inputInteraction(model,vocloader,txtpr,token,idxloader,padder)