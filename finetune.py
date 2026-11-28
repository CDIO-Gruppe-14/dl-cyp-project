#!/usr/bin/env python3
"""Fine-tune one model (Atom-JEPA or a baseline) for regression OR classification.

Configured with Hydra (conf/), e.g. python finetune.py model=morgan_mlp task=classification"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import matthews_corrcoef, mean_absolute_error, r2_score, roc_auc_score, average_precision_score
from models import MODELS
from models.gbm import LightGBMModel
from prepare_data import REG, CLS

import matplotlib.pyplot as plt #vi importerer matplotlib.pyplot for at kunne lave plots af training loss og validerings metrics


def masked_loss(pred, y, task, mean, std):
    mask = torch.isfinite(y)
    if not mask.any():
        return None
    # Index BEFORE applying the loss: multiplying NaN losses by zero is not safe.
    if task == 'regression':
        return nn.functional.mse_loss(pred[mask], ((y - mean) / std)[mask])
    return nn.functional.binary_cross_entropy_with_logits(pred[mask], y[mask])


def metrics(frame, predictions, targets, task):
    result = {}
    for j, target in enumerate(targets):
        y = frame[target].to_numpy(dtype=float)
        mask = np.isfinite(y)
        y, pred = y[mask], predictions[mask, j]
        if not len(y):
            result[target] = {'n': 0, ('ST-RAE' if task == 'regression' else 'MCC'): None}
            continue
        if task == 'regression':
            low = frame[target + '_conf_low'].to_numpy(dtype=float)[mask]
            high = frame[target + '_conf_high'].to_numpy(dtype=float)[mask]
            if not (np.isfinite(low).all() and np.isfinite(high).all() and (low <= high).all()):
                raise ValueError(f'Invalid credible intervals for {target}')
            distance = lambda p: np.maximum(low - p, 0) + np.maximum(p - high, 0)
            denom = distance(y.mean()).sum()
            result[target] = {'MAE': float(mean_absolute_error(y, pred)),
                              'R2': float(r2_score(y, pred)) if len(y) > 1 else None,
                              'ST-RAE': float(distance(pred).sum() / denom) if denom > 0 else None,
                              'n': len(y)}
        else:
            both = len(np.unique(y)) == 2
            result[target] = {'MCC': float(matthews_corrcoef(y, pred >= .5)),
                              'AUROC': float(roc_auc_score(y, pred)) if both else None,
                              'AP': float(average_precision_score(y, pred)) if both else None,
                              'n': len(y), 'positive_fraction': float(y.mean())}
    primary = 'ST-RAE' if task == 'regression' else 'MCC'
    scores = [result[t][primary] for t in targets]
    # Do not silently omit an endpoint when reporting a macro score.
    result['macro_' + primary] = (float(np.mean(scores))
        if all(s is not None and np.isfinite(s) for s in scores) else None)
    return result


@torch.no_grad()
def predict(model, loader, task, mean, std):
    model.eval()
    out = []
    for batch in loader:
        p = model(batch)
        p = p * std + mean if task == 'regression' else p.sigmoid()
        out.append(p.float().cpu().numpy())
    values = np.concatenate(out)
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite predictions')
    return values


def train_torch(cfg, train, val, targets, task, data, out):
    """Trains a torch model with the epoch loop; returns a function frame -> predictions."""
    if cfg.train.epochs < 1 or cfg.train.batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    model = MODELS[cfg.model.type](cfg.model, cfg.readout, len(targets), data, cfg.device)
    model.fit_preprocessing(train)  # Feature scaling etc. is fitted on the training split ONLY.
    mean = torch.zeros(len(targets), device=model.device)
    std = torch.ones_like(mean)
    if task == 'regression':
        # Fit scaling on the training fold ONLY, separately for each observed endpoint.
        mean = torch.tensor(train[targets].mean().to_numpy(), dtype=torch.float32, device=model.device)
        std = torch.tensor(train[targets].std(ddof=0).to_numpy(), dtype=torch.float32,
                           device=model.device).clamp_min(1e-6)
    batch_size = cfg.train.batch_size
    train_loader = model.loader(train, targets, batch_size, shuffle=True)
    val_loader = model.loader(val, targets, batch_size)
    optimizer = torch.optim.AdamW(model.param_groups(), weight_decay=cfg.train.weight_decay)
    best = float('inf')

    #opretter lister til at gemme training loss og validerings metrics for hver epoch
    train_losses = []
    val_metrics = []

    #Early stopping parameters
    patience = 20  # Number of epochs to wait for improvement before stopping
    counter = 0  # Counter for epochs without improvement

    print("batch_size: {}, epochs: {}, weight_decay: {}".format(batch_size, cfg.train.epochs, cfg.train.weight_decay))
    print("hidden_dim: {}, num_layers: {}, dropout: {}".format(cfg.readout.hidden_dim, cfg.readout.num_layers, cfg.readout.dropout))

    # Begynder træning - epoch loop
    for epoch in range(1, cfg.train.epochs + 1):

        # Nedstående er til at initially freeze så vi åbner for encoder efter 5 epochs:
        if cfg.model.type in ['atom_jepa', 'atom_jepa_class_freeze_e15', 'atom_jepa_regress_2', 'atom_jepa_regress_3'] and cfg.model.freeze_encoder and epoch == 21 :
            #efter 6 epochs unfreezer vi encoderens parametre, så de nu kan opdateres under træning. Vi sætter også grad checkpointing til igen:
            model.encoder.freeze_encoder = False  # Unfreeze encoder efter 20 epochs
            model.encoder.set_grad_checkpointing(True)  # sætter nu grad checkpointing til igen
            model.encoder.requires_grad_(True)
            print("\nUnfreezing encoder parameters after 5 epochs. Now training both encoder and readout head.\n")


            # Opdaterer optimizer til også at opdaterer encoderens parametre med en separat learning rate:
            optimizer = torch.optim.AdamW(model.param_groups(), weight_decay=cfg.train.weight_decay)
            optimizer.add_param_group({'params': list(model.encoder.parameters()), 'lr': cfg.model.encoder_lr})


        model.train() #sætter model i træningstilstand
        loss_sum, observed = 0., 0 # initialiserer summen af loss og antal observerede samples

        for batch in train_loader: # gennemgår hver batch i training loaderen
            y = batch['y'].to(model.device)

            # laver forward pass og beregner loss
            pred = model(batch).float()
            loss = masked_loss(pred, y, task, mean, std) #vi beregner loss for batchen, men kun for de samples hvor y er finite (ikke NaN)

            #error checks:
            if loss is None: # hvis der ikke er nogen observerede samples i batchen, fortsæt til næste batch
                continue
            if not torch.isfinite(loss): 
                raise RuntimeError('Nonfinite training loss')

            # laver nu backpropagation og opdaterer parametre:
            optimizer.zero_grad(set_to_none=True) #nulstiller gradienter
            loss.backward() #laver backpropagation
            nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']],
                                     cfg.train.grad_clip) #klipper gradienter - regulazation
            optimizer.step() #opdaterer parametre

            n = int(torch.isfinite(y).sum()) # tæller antallet af observerede samples i batchen (de hvor y er finite)
            loss_sum += loss.item() * n 
            observed += n
        
        score = metrics(val, predict(model, val_loader, task, mean, std), targets, task)
        primary = score['macro_ST-RAE' if task == 'regression' else 'macro_MCC']
        if primary is None:
            raise ValueError('Validation metric undefined; check endpoint coverage and split')
        selection = primary if task == 'regression' else -primary
        record = {'epoch': epoch, 'train_loss': loss_sum / observed, **score}

        # Gemmer training loss og validerings metrics for hver epoch
        train_losses.append(loss_sum / observed) # dividerer loss_sum med observed for at få gennemsnitlig training loss
        val_metrics.append(score) # gemmer validerings metrics for hver epoch

        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if selection < best:
            best = selection
            counter = 0  # reset counter if we have a new best score
            torch.save({'model': model.state_dict(), 'mean': mean, 'std': std,
                        'epoch': epoch, 'config': cfg_dict(cfg)}, out / 'best.pt')
        else:
            counter += 1  # increment counter if no improvement
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    # Gemmer training loss og validerings metrics som plots
    # Plot training loss og validation metric
    val_scores = [m['macro_ST-RAE' if task == 'regression' else 'macro_MCC'] for m in val_metrics]
    epochs = range(1, len(train_losses) + 1)

    #plotter training loss og validation metric i graf:
    plt.figure()
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.plot(epochs, val_scores, 'b', label='Validation ST-RAE' if task == 'regression' else 'Validation MCC')
    plt.axvline(saved_epoch, color='gray', linestyle='--') if False else None
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training vs Validation')
    plt.show()
    
    saved = torch.load(out / 'best.pt', map_location=model.device, weights_only=True)
    model.load_state_dict(saved['model'])
    mean, std = saved['mean'], saved['std']
    print(f'Best epoch: {saved["epoch"]}', flush=True)
    return lambda frame: predict(model, model.loader(frame, targets, batch_size), task, mean, std)


def cfg_dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def main(cfg):
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        cfg.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
    if cfg.task not in ('regression', 'classification'):
        raise ValueError(f'Unknown task {cfg.task!r}')
    out = Path(HydraConfig.get().runtime.output_dir)
    data = Path(cfg.data)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    task = cfg.task
    targets = REG if task == 'regression' else CLS
    train_all = pd.read_csv(data / task / 'train.csv')
    val_all = pd.read_csv(data / task / 'val.csv')
    frame = pd.concat([train_all, val_all], ignore_index=True)
    train = train_all[train_all[targets].notna().any(axis=1)].copy()
    val = val_all[val_all[targets].notna().any(axis=1)].copy()
    if not len(train) or not len(val):
        raise ValueError('Empty train or validation set')
    if set(train_all.canonical_smiles) & set(val_all.canonical_smiles):
        raise ValueError('Exact molecule leakage between train and validation')
    for t in targets:
        if not train[t].notna().any() or not val[t].notna().any():
            raise ValueError(f'Missing train/validation labels for {t}')
        if task == 'classification':
            if not set(frame[t].dropna()).issubset({0, 1}):
                raise ValueError(f'Nonbinary labels: {t}')
            if train[t].nunique() != 2 or val[t].nunique() != 2:
                raise ValueError(f'{t} needs both classes in train and validation; revise split')
    config = cfg_dict(cfg)
    packages = ['torch', 'atom-jepa', 'rdkit', 'numpy', 'pandas', 'scikit-learn', 'hydra-core',
                'descriptastorus', 'lightgbm']
    config.update(targets=targets, output_dir=str(out),
                  versions={k: importlib.metadata.version(k) for k in packages},
                  preparation=json.loads((data / 'manifest.json').read_text()),
                  split_sha256={name: hashlib.sha256((data / task / f'{name}.csv').read_bytes()).hexdigest()
                                for name in ['train', 'val', 'test']})
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    frame[['Molecule_Name', 'key', 'split']].to_csv(out / 'split.csv', index=False)

    # Sanity baseline fitted on training labels only.
    baseline = np.tile(train[targets].mean().to_numpy(), (len(val), 1))
    (out / 'baseline_metrics.json').write_text(json.dumps(metrics(val, baseline, targets, task), indent=2))
    if cfg.model.type == 'lightgbm':
        model = LightGBMModel(cfg.model, data, cfg.seed)
        best = model.fit(train, val, targets, task, out)
        (out / 'best_iterations.json').write_text(json.dumps(best, indent=2))
        predict_frame = model.predict
    else:
        predict_frame = train_torch(cfg, train, val, targets, task, data, out)
    validation_predictions = predict_frame(val)
    (out / 'validation_metrics.json').write_text(json.dumps(metrics(val, validation_predictions, targets, task), indent=2))
    vp = val[['SMILES', 'Molecule_Name']].copy()
    vp[targets] = validation_predictions
    vp.to_csv(out / 'validation_predictions.csv', index=False)
    # Only now load and evaluate the held-out labelled test split.
    test = pd.read_csv(data / task / 'test.csv')
    if not len(test):
        raise ValueError('Empty test set')
    if set(frame.canonical_smiles) & set(test.canonical_smiles):
        raise ValueError('Exact molecule leakage into the test split')
    predictions = predict_frame(test)
    test_scores = metrics(test, predictions, targets, task)
    (out / 'test_metrics.json').write_text(json.dumps(test_scores, indent=2))
    # Classification exports probabilities; metrics use a fixed 0.5 threshold.
    output = test[['SMILES', 'Molecule_Name']].copy()
    output[targets] = predictions
    output.to_csv(out / 'test_predictions.csv', index=False)
    print(f'Held-out test metrics: {test_scores}')

if __name__ == '__main__':
    main()
