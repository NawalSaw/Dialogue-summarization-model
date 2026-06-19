from pathlib import Path

def get_config():
    return {
        'lr': 3e-4, 
        "d_model": 512,
        "heads_num": 8,
        "dropout": 0.1,
        "num_layers": 6,
        "src_seq_len": 320,
        "tgt_seq_len": 72,
        "batch_size": 32,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        'num_epochs': 25,
        "tokenizer_path": "models/tokenizer.json",
        "experiment_name": "runs/tmodel",
        "warmup_ratio": 0.10,
        "patience": 5
    }

def get_weights_path(config, epoch: int):
    return str(Path(config['model_folder']) / f"{config['model_basename']}{epoch:02d}.pt")
