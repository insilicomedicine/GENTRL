import torch

from torch.utils.data import Dataset

import pandas as pd
import numpy as np


class MolecularDataset(Dataset):
    def __init__(self, sources=[], props=['logIC50', 'BFL', 'pipeline'],
                 with_missings=False):
        self.num_sources = len(sources)

        self.source_smiles = []
        self.source_props = []
        self.source_missings = []
        self.source_probs = []

        self.with_missings = with_missings

        self.len = 0
        for source_descr in sources:
            cur_df = pd.read_csv(source_descr['path'])
            cur_smiles = list(cur_df[source_descr['smiles']].values)

            cur_props = torch.zeros(len(cur_smiles), len(props)).float()
            cur_missings = torch.zeros(len(cur_smiles), len(props)).long()

            for i, prop in enumerate(props):
                if prop in source_descr:
                    if isinstance(source_descr[prop], str):
                        cur_props[:, i] = torch.from_numpy(
                            cur_df[source_descr[prop]].values)
                    else:
                        cur_props[:, i] = torch.from_numpy(
                            cur_df[source_descr['smiles']].map(
                                source_descr[prop]).values)
                else:
                    cur_missings[:, i] = 1

            self.source_smiles.append(cur_smiles)
            self.source_props.append(cur_props)
            self.source_missings.append(cur_missings)
            self.source_probs.append(source_descr['prob'])

            self.len = max(self.len,
                           int(len(cur_smiles) / source_descr['prob']))

        self.source_probs = np.array(self.source_probs).astype(np.float)

        self.source_probs /= self.source_probs.sum()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        trial = np.random.random()

        s = 0
        for i in range(self.num_sources):
            if (trial >= s) and (trial <= s + self.source_probs[i]):
                bin_len = len(self.source_smiles[i])
                sm = self.source_smiles[i][idx % bin_len]

                props = self.source_props[i][idx % bin_len]
                miss = self.source_missings[i][idx % bin_len]

                if self.with_missings:
                    return sm, torch.concat([props, miss])
                else:
                    return sm, props

            s += self.source_probs[i]
