from pathlib import Path

def get_config():
    return {
        'lr': 1e-4, 
        "d_model": 512,
        "heads_num": 4,
        "dropout": 0.2,
        "num_layers": 3,
        "seq_len": 1400,
        "batch_size": 8,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        'num_epochs': 10,
        "tokenizer_path": "models/tokenizer.json",
        "experiment_name": "runs/tmodel",
    }

def get_weights_path(config, epoch: int):
    return str(Path(config['model_folder']) / f"{config['model_basename']}{epoch:02d}.pt")
