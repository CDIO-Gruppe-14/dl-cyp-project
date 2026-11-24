"""Pretrained ChemBERTa-2 SMILES encoder with a readout head from models/readouts.py"""
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from models.readouts import build_readout

class Smiles(Dataset):
    """Én række = (SMILES-streng, labels)"""
    def __init__(self, frame, targets):
        self.smiles = frame.SMILES.tolist()
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles[i], torch.tensor(self.y[i])

class ChemBERTaModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
        self.encoder = AutoModel.from_pretrained(cfg.checkpoint).to(self._device)
        self.encoder.requires_grad_(not cfg.freeze_encoder)
        self.readout = build_readout(readout, self.encoder.config.hidden_size, n_targets).to(self._device)

    @property
    def device(self):
        return self._device

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    def collate(self, items):
         smiles, y = zip(*items)
         encoded = self.tokenizer(list(smiles), padding=True, truncation=True,
                                  max_length=self.cfg.max_length, return_tensors='pt')
         return {'input_ids': encoded['input_ids'],
                 'attention_mask': encoded['attention_mask'],
                 'y': torch.stack(y)}

    def loader(self, frame, targets, batch_size, shuffle=False):
        return DataLoader(Smiles(frame, targets), batch_size=batch_size, 
                          shuffle=shuffle,collate_fn=self.collate, num_workers=0)

    def param_groups(self):
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder:
            groups.append({'params': list(self.encoder.parameters()), 'lr': self.cfg.encoder_lr})
        return groups

    def forward(self, batch):
        ids = batch['input_ids'].to(self._device)
        mask = batch['attention_mask'].to(self._device)
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.cfg.pooling == 'cls':
            embedding = hidden[:, 0]
        elif self.cfg.pooling == 'mean':
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            embedding = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1e-6)
        else:
            raise ValueError(f'Unknown pooling {self.cfg.pooling!r}')
        return self.readout(embedding)