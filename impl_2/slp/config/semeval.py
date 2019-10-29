from enum import Enum


TASK7_URL = 'https://www.cs.rochester.edu/u/nhossain/humicroedit/semeval-2020-task-7-data.zip'  # noqa
TASK11_URL = 'https://propaganda.qcri.org/semeval2020-task11/data/datasets.tgz'
TASK11_TOOLS = 'https://propaganda.qcri.org/semeval2020-task11/data/tools.tgz'


class SPECIAL_TOKENS(Enum):
    PAD = '[PAD]'
    MASK = '[MASK]'
    UNK = '[UNK]'
    BOS = '[BOS]'
    EOS = '[EOS]'
    CLS = '[CLS]'
    UNFUNNY = '[UNFUNNY]'
    FUNNY = '[FUNNY]'

    @classmethod
    def has_token(cls, token):
        return any(token == t.name or token == t.value
                   for t in cls)

    @classmethod
    def to_list(cls):
        return list(map(lambda x: x.value, cls))
