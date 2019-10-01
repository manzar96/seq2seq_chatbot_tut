import gensim
import numpy as np

class WordEmbLoader:
    """
    this class returns the word embeddings asked(glove,word2vec -gensim,fasttext
    etc).
    """
    def load_embeddings(self,embfile,selected="glove"):
        if selected=="glove":
            return self.load_glove(embfile)
        elif selected=="gensim":
            return self.load_gensim(embfile)

    def load_glove(self,embfile):
        words = []
        idx = 0
        word2idx = {}
        idx2word = {}
        # vectors = bcolz.carray(np.zeros(1), rootdir=f'{embfile}/6B.50.dat',
        #                        mode='w')
        vectors = []

        with open(embfile, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)

        return word2idx, idx2word, np.asarray(vectors)



    def load_gensim(self,embfile):
        model = gensim.models.KeyedVectors.load_word2vec_format(embfile,
                                                                limit=5000,
                                                                binary=True)
        word2idx = {}
        idx2word = {}
        idx2embedding = {}
        embeddings = []
        for index, word in enumerate(model.vocab):
            word2idx[word] = index
            idx2word[index] = word
            embeddings.append(model[word])
        return word2idx, idx2word, np.asarray(embeddings)


