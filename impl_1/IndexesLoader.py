

class IndexesLoader:
    """
    This class maps text to indexes.
    It adds index of end_token at the end of each turn.
    If word is not found it replaces it with index of unk token.
    """
    def __init__(self, voc, unk_token, end_token=None):
        self.voc = voc
        self.end_token = end_token
        self.unk_token = unk_token

    def indexesFromSentence(self, sentence):
        idxlist = []
        if self.end_token is not None:

            for word in sentence:
                if word in self.voc.word2idx.keys():
                    idxlist.append(self.voc.word2idx[word])
                else:
                    idxlist.append(self.unk_token)
            idxlist.append(self.end_token)
        else:
            for word in sentence:
                if word in self.voc.word2idx.keys():
                    idxlist.append(self.voc.word2idx[word])
                else:
                    idxlist.append(self.unk_token)
        return idxlist

    def get_indexes(self, sentences):

        indexes_list = [self.indexesFromSentence(sentence) for
                        sentence in sentences]

        return indexes_list


