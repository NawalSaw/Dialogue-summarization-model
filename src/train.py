import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from pathlib import Path

from dataset import SamsumDataset, causal_mask
from tokeniser.tokeniser import get_or_create_tokenizer
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from transformer import build_transformer
from config import get_config
from config import get_weights_path

def greedy_decode(model, tokenizer, source, source_mask, src_seq_len, tgt_seq_len, device):
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    
    # Precomputing
    encoder_output = model.encode(source, source_mask)
    
    # Start with BOS token
    decoder_input = torch.tensor([[bos_token_id]], dtype=torch.int64, device=device) # shape (1, 1)
    
    # Generate tokens
    while True:
        if decoder_input.size(1) >= tgt_seq_len:
            break

        # Prepare decoder mask
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Get decoder output
        decoder_output = model.decode(decoder_input, encoder_output, source_mask, decoder_mask) # shape (1, seq_len, d_model)
        
        # Get next token
        next_token_probs = model.project(decoder_output[:, -1, :]) # shape (1, vocab_size)
        _, next_token = torch.max(next_token_probs, dim=-1, keepdim=True) # shape (1, 1)
        
        # Append to decoder input
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1, dtype=torch.int64, device=device).type_as(source).fill_(next_token.item())], dim=1)
    
        # Check for EOS
        if next_token.item() == eos_token_id:
            break
            
    return decoder_input.squeeze(0)
    
def evaluate_model(model, tokenizer, src_seq_len, tgt_seq_len, validation_dataset, device, print_msg, num_example=2):
    model.eval()
    count = 0

    console_width = 80

    with torch.no_grad():
        for batch in validation_dataset:
            if count >= num_example:
                break
            count += 1

            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for evaluation"

            model_out = greedy_decode(model, tokenizer, encoder_input, encoder_mask, src_seq_len, tgt_seq_len, device)
            
            source_text = batch["src_text"][0]
            target_text = batch["trg_text"][0]
            model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())

            print_msg(f"{'SOURCE':>12}: {source_text}")
            print_msg(f"{'TARGET':>12}: {target_text}")
            print_msg(f"{'PREDICTED':>12}: {model_out_text}")

            print_msg("=" * console_width)
            
def get_dataset_tokenizer(config):
    dataset = load_dataset("knkarthick/samsum")

    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    com_dataset = concatenate_datasets([train_dataset, val_dataset, test_dataset])
    tokenizer = get_or_create_tokenizer(config, com_dataset, config["batch_size"])

    ### Filtering ###
    def keep_example(example):
        src_len = len(tokenizer.encode(example["dialogue"]).ids)
        tgt_len = len(tokenizer.encode(example["summary"]).ids)

        return src_len <= 481 and tgt_len <= 65
    
    filtered_train = train_dataset.filter(keep_example)
    filtered_val = val_dataset.filter(keep_example)
    filtered_test = test_dataset.filter(keep_example)

    train_dataset = SamsumDataset(filtered_train, tokenizer, config["src_seq_len"], config["tgt_seq_len"])
    val_dataset = SamsumDataset(filtered_val, tokenizer, config["src_seq_len"], config["tgt_seq_len"])
    test_dataset = SamsumDataset(filtered_test, tokenizer, config["src_seq_len"], config["tgt_seq_len"])

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer

def get_model(config, vocab_size):
    model = build_transformer(config["d_model"], config["heads_num"], config["dropout"], vocab_size, config["num_layers"], config["src_seq_len"], config["tgt_seq_len"])
    return model

# 3. Define the scheduling mathematical logic
def lr_lambda(current_step: int, warmup_steps: int, total_training_steps: int):
    # Phase 1: Linear Warmup
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    
    # Phase 2: Cosine Decay down to 10% of peak LR
    progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
    import math
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(0.1, cosine_decay) # Prevents LR from dropping all the way to absolute 0

def train_model(config):
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader, tokenizer = get_dataset_tokenizer(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    writer = SummaryWriter(config["experiment_name"])

    # Calculate steps
    num_batches_per_epoch = len(train_dataloader)
    total_training_steps = config["num_epochs"] * num_batches_per_epoch
    warmup_steps = int(total_training_steps * config["warmup_ratio"])

    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"], eps=1e-9, weight_decay=0.01)
    scheduler = LambdaLR(optimiser, lambda current_step: lr_lambda(current_step, warmup_steps, total_training_steps))

    print(f"Total training steps: {total_training_steps}")
    print(f"Warmup steps: {warmup_steps}")

    intial_epoch = 0
    global_step = 0
                
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    print(f"vocabulary size: {tokenizer.get_vocab_size()}")

    if config['preload']:
        model_filename = get_weights_path(config, config['preload'])
        print(f"Preloading model {model_filename}")

        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])

        scheduler.load_state_dict(state['scheduler_state_dict'])
        print("Preloaded scheduler state successfully.")

        intial_epoch = state['epoch'] + 1
        optimiser.load_state_dict(state['optimizer_state_dict'])

        global_step = state['global_step']
        print(f"Preloaded model from epoch {intial_epoch - 1}")
        print(f"Global step: {global_step}")

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(intial_epoch, config["num_epochs"]):
        model.train()
        batch_iter = tqdm(train_dataloader, desc=f"Epoch {epoch}")

        for batch in batch_iter:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, vocab_size)
            
            loss = loss_fn(proj_output.reshape(-1, tokenizer.get_vocab_size()), label.reshape(-1))

            writer.add_scalar('train_loss', loss.item(), global_step)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

            global_step += 1

            # NEW: Track current learning rate in TensorBoard to verify it works
            current_lr = optimiser.param_groups[0]["lr"]
            writer.add_scalar('learning_rate', current_lr, global_step)
            
            # Optional: Display current LR in progress bar
            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}", "lr": f"{current_lr:.2e}"})
            writer.flush()

        checkpoint_path = get_weights_path(config, epoch)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), # Add this
            "global_step": global_step
        }, checkpoint_path)

        evaluate_model(model, tokenizer, config["src_seq_len"], config["tgt_seq_len"], val_dataloader, device, lambda x: batch_iter.write(x))

    model_path = get_weights_path(config, epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step
    }, model_path)

    writer.close()
    print(f"Model saved to {model_path}")
    

if __name__ == "__main__":
    config = get_config()
    train_model(config)
