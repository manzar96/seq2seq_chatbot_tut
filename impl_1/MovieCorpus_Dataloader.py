class MovieCorpusDataloader:

    def __init__(self):
        self.questions = []
        self.answers = []

    def load_data(self, movie_lines_path, movie_convs_path):

        movie_lines = open(movie_lines_path, encoding='utf-8',
                           errors='ignore').read().split('\n')
        movie_conv_lines = open(movie_convs_path, encoding='utf-8',
                                errors='ignore').read().split('\n')

        # Create a dictionary to map each line's id with its text
        id2line = {}
        for line in movie_lines:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]

        # Create a list of all of the conversations lines ids.
        convs = []
        for line in movie_conv_lines[:-1]:
            _line = line.split(' +++$+++ ')[-1][1:-1]\
                .replace("'", "").replace(" ", "")
            convs.append(_line.split(','))

        # Sort the sentences into questions (inputs) and answers (targets)
        questions = []
        answers = []

        for conv in convs:
            for i in range(len(conv) - 1):
                questions.append(id2line[conv[i]])
                answers.append(id2line[conv[i + 1]])

        self.questions = questions
        self.answers = answers
        return self.questions, self.answers
