import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
"""
This class processes text.
During init why should select the basic processes that should be done
e.g. convert to lowercase: lowercase=True

Process text function receives a list of texts that should be processed.
It returns the list of texts processed.
"""


class TextProcessor:

    def __init__(self, lowercase, cleaning=False, remove_punct=False,
                 remove_stopwords=False, thresholding=False, threshold_min=2,
                 threshold_max=25):

        self.lowercase = lowercase
        self.thresholding = thresholding
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.cleaning = cleaning
        self.remove_stop = remove_stopwords
        self.remove_punct = remove_punct
        self.ntlk_stop_words = set(stopwords.words('english'))

    def clean_line(self, text):

        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", text)
        return text

    def remove_punctuation(self, text):

        text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
        return text

    def remove_stopwords(self, text):

        word_tokens = word_tokenize(text)

        filtered_sentence = [w for w in word_tokens if w not in
                             self.ntlk_stop_words]

        filtered_text = ""
        for w in filtered_sentence:
            filtered_text = filtered_text+" "+w
        return filtered_text

    """
    Each element of data list includes some lines of text. 
    """
    def process_text(self, data_list):

        new_data = data_list

        if self.lowercase:
            print('\n>>>Converting letters to lowercase...')
            new_data = [x.lower() for x in data_list]

        if self.cleaning:
            print('\n>>>Cleaning...')
            new_data = [self.clean_line(line) for line in new_data]

        if self.remove_punct:
            print("\n>>>Removing punctuation")
            new_data = [self.remove_punctuation(line) for line in new_data]

        if self.remove_stop:
            print("\n>>>Removing stopwords")
            new_data = [self.remove_stopwords(line) for line in new_data]

        return new_data




