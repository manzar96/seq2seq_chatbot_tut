import numpy as np
import torch
import torch.nn as nn
from ignite.metrics import Loss
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from slp.util.embeddings import EmbeddingsLoader
from slp.data.movies_corpus_dataset import MovieCorpusDataset
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.data.collators import Seq2SeqCollator
from slp.trainer.trainer import Seq2SeqTrainer

from slp.modules.rnn import EncoderDecoder,Encoder,Decoder

from slp.modules.loss import RMSELoss
from torch.optim import Adam

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLLATE_FN = Seq2SeqCollator(device='cpu')
KFOLD = False
MAX_EPOCHS = 50


def dataloaders_from_indices(dataset, train_indices, val_indices, batch_train,
                             batch_val):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=COLLATE_FN)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=COLLATE_FN)
    return train_loader, val_loader


def train_test_split(dataset, batch_train, batch_val,
                     test_size=0.2, shuffle=True, seed=None):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[test_split:]
    val_indices = indices[:test_split]
    return dataloaders_from_indices(train_indices, val_indices, batch_train, batch_val)


def kfold_split(dataset, batch_train, batch_val, k=5, shuffle=True, seed=None):
    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
    for train_indices, val_indices in kfold.split(dataset):
        yield dataloaders_from_indices(train_indices, val_indices, batch_train, batch_val)

def trainer_factory(embeddings,device=DEVICE):
    encoder = Encoder(hidden_size=254, embeddings=embeddings, device=DEVICE)
    decoder = Decoder(max_target_len=14, hidden_size=254,
                      embeddings=embeddings, device=DEVICE)

    model = EncoderDecoder(encoder, decoder, device=DEVICE)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    metrics = {
        'loss': Loss(criterion)
    }

    trainer = Seq2SeqTrainer(model,
                            optimizer,
                            checkpoint_dir=None, # '../checkpoints',
                            metrics=metrics,
                            non_blocking=True,
                            retain_graph=True,
                            patience=5,
                            device=device,
                            loss_fn=criterion)
    return trainer


if __name__ == '__main__':
    loader = EmbeddingsLoader(
        '../cache/glove.6B.50d.txt', 50)
    word2idx, _, embeddings = loader.load()

    tokenizer = SpacyTokenizer()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device='cpu')

    dataset = MovieCorpusDataset('../../data/', transforms=[], train=True)
    dataset = dataset.map(tokenizer).map(to_token_ids).map(to_tensor)

    if KFOLD:
        cv_scores = []
        import gc
        for train_loader, val_loader in kfold_split(dataset, 32, 128):
            trainer = trainer_factory(embeddings, device=DEVICE)
            fold_score = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)
            cv_scores.append(fold_score)
            del trainer
            gc.collect()
        final_score = float(sum(cv_scores)) / len(cv_scores)
    else:
        train_loader, val_loader = train_test_split(dataset, 32, 128)
        trainer = trainer_factory(embeddings, device=DEVICE)
        final_score = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)

    print(f'Final score: {final_score}')