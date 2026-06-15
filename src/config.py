from pathlib import Path

def get_config():
    return {
        'lr': 3e-4, 
        "d_model": 256,
        "heads_num": 4,
        "dropout": 0.1,
        "num_layers": 4,
        "src_seq_len": 512 + 2, # +2 for [BOS] and [EOS]
        "tgt_seq_len": 80 + 1, # +1 for [EOS]
        "batch_size": 8,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        'num_epochs': 20,
        "tokenizer_path": "models/tokenizer.json",
        "experiment_name": "runs/tmodel",
        "warmup_ratio": 0.10
        
    }

def get_weights_path(config, epoch: int):
    return str(Path(config['model_folder']) / f"{config['model_basename']}{epoch:02d}.pt")
