from nltk.tokenize import word_tokenize


class TextTokenizer:

    """
    This function receives a list of texts and does word tokenization to
    each element of the list.It returns a list of lists!
    """
    def word_tokenization(self, texts_list):

        tokenized_texts = [word_tokenize(text) for text in texts_list]
        return tokenized_texts
