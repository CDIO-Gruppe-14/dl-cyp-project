"""Pretrained Atom-JEPA encoder with a readout head from models/readouts.py."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from atom_jepa import AtomJEPA
from models.readouts import build_readout

# Indlæser og forbereder data til træning og validering
class Conformers(Dataset):
    def __init__(self, frame, folder, targets): 
        self.frame, self.folder = frame.reset_index(drop=True), folder
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)

    # Returnerer længden af datasættet
    def __len__(self):
        return len(self.frame)

    # Henter et element fra datasættet baseret på indekset i
    def __getitem__(self, i):
        with np.load(self.folder / f'{self.frame.iloc[i].key}.npz') as d:
            return {'atomic_numbers': torch.tensor(d['z'], dtype=torch.long),
                    'node_coordinates': torch.tensor(d['pos'], dtype=torch.float32),
                    'y': torch.tensor(self.y[i])}

# Atom-JEPA model med et readout head for regression eller classification tasks
class AtomJEPAModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device): # 
        super().__init__()
        self.cfg, self.folder = cfg, data_dir / 'conformers'
        self.encoder = AtomJEPA.from_pretrained(cfg.checkpoint, device=device, # Indlæser den pretrained Atom-JEPA encoder
                                                precision=cfg.precision, cuequivariance=False)
        # Indstiller om encoderens parametre skal opdateres under træning, afhængigt af konfigurationen
        self.encoder.set_grad_checkpointing(not cfg.freeze_encoder)
        self.encoder.requires_grad_(not cfg.freeze_encoder)
        # Bygger readout head baseret på konfigurationen. Får encoderens embedding som input og giver prediction ud fra dette
        self.readout = build_readout(readout, self.encoder.embedding_dim, n_targets).to(self.encoder.device)

    @property
    def device(self):
        return self.encoder.device #sætter bare device altså cpu eller gpu

    # vi kan vælge om vi vil træne hele modellen eller kun readout head'en.
    # Hvis freeze_encoder er True, vil encoderens parametre ikke blive opdateret under træning:
    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval() # Holder encoder i evaluation mode
        return self

    def fit_preprocessing(self, train):
        pass

    # Indlæser molekyler i batches til træning/evaluering og returnerer en DataLoader. Bruges i finetune.py
    def loader(self, frame, targets, batch_size, shuffle=False):
        return DataLoader(Conformers(frame, self.folder, targets), batch_size=batch_size,
                          shuffle=shuffle, collate_fn=self.encoder.collate, num_workers=0)

    # Returnerer parametre grupper for optimering. Bruges i finetune.py
    def param_groups(self):
        # Readout og encoder kan have forskellige learning rates 
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder: # hvis 
            groups.append({'params': list(self.encoder.parameters()), 'lr': self.cfg.encoder_lr})
        return groups
    
    # Forward pass through the model
    def forward(self, batch):
        return self.readout(self.encoder(batch))  # Differentiable forward; do NOT use encoder.embed().
