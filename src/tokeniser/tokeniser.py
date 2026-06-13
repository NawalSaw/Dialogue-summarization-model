from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents

from pathlib import Path

def get_corpus(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        
        # Extract both text fields for this batch
        dialogues = batch["dialogue"]
        summaries = batch["summary"]
 
        # Combine the lists so they are fed sequentially, not concatenated as strings
        # This yields 2000 total independent text items per 1000 row batch
        yield dialogues + summaries

def get_or_create_tokenizer(config, dataset, batch_size=1000):
    tokenizer_path = Path(config['tokenizer_path'])
    
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], min_frequency=2)

        tokenizer.train_from_iterator(get_corpus(dataset, batch_size), trainer=trainer)

        print(f"Tokenizer trained and saved at {tokenizer_path}")
        tokenizer.save(str(tokenizer_path))

    else:
        print(f"Tokenizer loaded from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

