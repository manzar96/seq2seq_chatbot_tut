import csv
import os
import re
from zipfile import ZipFile

from torch.utils.data import Dataset

from slp.config.semeval import TASK7_URL
from slp.util.system import download_url


class Task1Dataset(Dataset):
    def __init__(self, directory, transforms=None, train=True):
        dest = download_url(TASK7_URL, directory)
        with ZipFile(dest, 'r') as zipfd:
            zipfd.extractall(directory)
        split = 'train' if train else 'dev'
        self._file = os.path.join(directory, 'data', 'task-1', f'{split}.csv')
        self.transforms = transforms
        (self.ids,
         self.original,
         self.edit,
         self.grades,
         self.mean_grade) = self.get_metadata(self._file)
        self.transforms = transforms

    def get_metadata(self, _file):
        rows = []
        with open(_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                rows.append(row)

        rows = rows[1:]  # strip header
        ids, original, edit, grades, mean_grade = zip(*rows)
        return ids, original, edit, grades, mean_grade

    def _edit(self, original, edit):
        return re.sub(r'<.*?/>', f'<{edit}/>', original)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        original = self.original[idx]
        edit = self.edit[idx]
        # ids = self.ids[idx]
        # grades = self.grades[idx]
        mean_grade = self.mean_grade[idx]
        edited = self._edit(original, edit)

        if self.transforms is not None:
            original = self.transforms(original)
            edited = self.transforms(edited)

        # grades = [int(g) for g in str(grades)]
        return original, edited, float(mean_grade)


if __name__ == '__main__':
    data = Task1Dataset('../../data/')
    for d in data:  # type: ignore
        print(d)
