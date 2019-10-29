from Word_Emb import WordEmbLoader
import numpy as np

class EmbVoc:
    """
    This class contains a dictionary mapping word to indexes and inverse. It
    also contains the embeddings of each word.
    pre_dict: is a dict containing custom embeddings.
    """
    def __init__(self, pre_dict=None):
        self.word2idx = {}
        self.idx2word = {}
        self.embeddings = []
        self.num_words = 0
        self.emb_dim = 0
        if pre_dict is not None:

            for key in pre_dict.keys():
                self.word2idx[key] = self.num_words
                self.idx2word[self.num_words] = key
                self.embeddings.append(pre_dict[key])
                self.num_words += 1
            self.emb_dim = self.embeddings[0].shape[0]

    def add_embeddings(self, embfile, sel="glove"):
        x = WordEmbLoader()
        w2i, i2w, emb = x.load_embeddings(embfile, selected=sel)

        assert emb.shape[1] == self.emb_dim , "Embedding dimensions don't match"
        for key in w2i.keys():
            self.word2idx[key] = self.num_words
            self.idx2word[self.num_words] = key
            self.embeddings.append(emb[w2i[key]])
            self.num_words += 1

        self.embeddings = np.asarray(self.embeddings)



    """
    This functions receives as input a list of words and returns a list with 
    the 
    equivalent indexes.
    """
    def get_indexes(self, input_list):
        return [self.word2idx[word] for word in input_list]

    """
    This functions receives as input a list of lists. Each one list is a 
    sentence splitted to words.
    """
    def get_indexes_from_sentences(self, sentences_list):
        return [self.get_indexes(sentence) for sentence in sentences_list]






