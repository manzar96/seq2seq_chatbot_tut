import itertools
import torch

class DataPadderMasker:
    def __init__(self, pad_token):
        self.pad_token = pad_token


    def zeroPadding(self, l,max_len):
        padded_inputs = []
        for utterance in l:
            if len(utterance)<max_len:
                new_utterance = utterance + [self.pad_token for i in range(
                    max_len - len(utterance))]
                padded_inputs.append(new_utterance)
            else:
                padded_inputs.append(utterance)
        return padded_inputs

    def binaryMatrix(self, l):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == self.pad_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    # Returns padded input sequence tensor and lengths
    def inputVar(self, input_data):
        lengths = torch.tensor([len(indexes) for indexes in input_data])
        max_len = max(lengths)
        padList = self.zeroPadding(input_data,max_len)
        padVar = torch.FloatTensor(padList)
        return padVar, lengths

    # Returns padded target sequence tensor, padding mask, and max target length
    def outputVar(self, input_data):
        max_target_len = max([len(indexes) for indexes in input_data])
        padList = self.zeroPadding(input_data, max_target_len)
        mask = self.binaryMatrix(padList)
        mask = torch.ByteTensor(mask)
        padVar = torch.FloatTensor(padList)
        return padVar, mask, max_target_len

    def pad_data(self, pairs):
        inputs, targets = [], []
        for pair in pairs:
            inputs.append(pair[0])
            targets.append(pair[1])
        inp, lengths = self.inputVar(inputs)
        output, mask, max_target_len = self.outputVar(targets)
        return inp, lengths, output, mask, max_target_len
