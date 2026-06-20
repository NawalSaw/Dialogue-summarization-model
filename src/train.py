import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from pathlib import Path

# Distributed Training Imports
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import SamsumDataset, causal_mask
from tokeniser.tokeniser import get_or_create_tokenizer
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from transformer import build_transformer
from config import get_config, get_weights_path

def greedy_decode(model, tokenizer, source, source_mask, src_seq_len, tgt_seq_len, device):
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    
    # Use unwrapped module when running inference inside DDP
    model_m = model.module if hasattr(model, "module") else model
    
    # Precomputing
    encoder_output = model_m.encode(source, source_mask)
    
    # Start with BOS token
    decoder_input = torch.tensor([[bos_token_id]], dtype=torch.long, device=device) # shape (1, 1)
    
    # Generate tokens
    while True:
        if decoder_input.size(1) >= tgt_seq_len:
            break

        # Prepare decoder mask
        decoder_mask = causal_mask(decoder_input.size(1)).unsqueeze(0).type_as(source_mask).to(device)        
        
        # Get decoder output
        decoder_output = model_m.decode(decoder_input, encoder_output, source_mask, decoder_mask) # shape (1, seq_len, d_model)
        
        # Get next token
        next_token_probs = model_m.project(decoder_output[:, -1, :]) # shape (1, vocab_size)
        _, next_token = torch.max(next_token_probs, dim=-1, keepdim=True) # shape (1, 1)
        
        # Append to decoder input
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1, dtype=torch.long, device=device).type_as(source).fill_(next_token.item())], dim=1)
    
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
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer.decode(model_out.detach().cpu().numpy())

            print_msg(f"{'SOURCE':>12}: {source_text}")
            print_msg(f"{'TARGET':>12}: {target_text}")
            print_msg(f"{'PREDICTED':>12}: {model_out_text}")
            print_msg("=" * console_width)
            
def get_dataset_tokenizer(config, rank, world_size):

    # Only let rank 0 download and tokenize dataset to avoid file race locks
    dataset=None
    if rank == 0:
        dataset = load_dataset("knkarthick/samsum")
    dist.barrier() # Wait until process 0 is done
    if rank != 0:
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

        return src_len <= 310 and tgt_len <= 65
        
    filtered_train = train_dataset.filter(keep_example)
    filtered_val = val_dataset.filter(keep_example)
    filtered_test = test_dataset.filter(keep_example)

    train_dataset = SamsumDataset(filtered_train, tokenizer, config["src_seq_len"], config["tgt_seq_len"])
    val_dataset = SamsumDataset(filtered_val, tokenizer, config["src_seq_len"], config["tgt_seq_len"])
    test_dataset = SamsumDataset(filtered_test, tokenizer, config["src_seq_len"], config["tgt_seq_len"])

    # Attach DistributedSamplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer

def get_model(config, vocab_size):
    model = build_transformer(config["d_model"], config["heads_num"], config["dropout"], vocab_size, config["num_layers"], config["src_seq_len"], config["tgt_seq_len"])
    return model

# Define the scheduling mathematical logic
def lr_lambda(current_step: int, warmup_steps: int, total_training_steps: int):
    # Phase 1: Linear Warmup
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    
    # Peak LR equals --> config["lr"]

    # Phase 2: Cosine Decay down to 10% of peak LR
    progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
    import math
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(0.01, cosine_decay) # Prevents LR from dropping all the way to absolute 0

def train_model(config): 
    # Initialize PyTorch Process Group for multi-GPU
    dist.init_process_group(backend="nccl")

    # Get environment variables managed by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set the local GPU device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Define a printing manager that isolates stdout to Main GPU (Rank 0)
    def print_msg(msg):
        if rank == 0:
            print(msg)

    print_msg(f"Running DDP on {world_size} GPUs. Current GPU Local Rank: {local_rank}")

    if rank == 0:
        Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, _, tokenizer = get_dataset_tokenizer(config, rank, world_size)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    # Wrap model with Distributed Data Parallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # TensorBoard setup only on Main GPU
    writer = None
    if rank == 0:
        writer = SummaryWriter(config["experiment_name"])
    
    # Calculate steps
    num_batches_per_epoch = len(train_dataloader)
    total_training_steps = config["num_epochs"] * num_batches_per_epoch
    warmup_steps = int(total_training_steps * config["warmup_ratio"])

    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"], eps=1e-5, weight_decay=0.01)
    scheduler = LambdaLR(optimiser, lambda current_step: lr_lambda(current_step, warmup_steps, total_training_steps))

    print_msg(f"Total training steps: {total_training_steps}")
    print_msg(f"Warmup steps: {warmup_steps}")

    intial_epoch = 0
    global_step = 0

    best_loss = float('inf')
    patience_counter = 0
    accumulation_steps = 8   # if batch_size=16, total effective batch = 16 * 2 GPUs * 2 = 64

    # Initialize AMP Scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda')
                
    num_params = sum(p.numel() for p in model.parameters())
    print_msg(f"Number of parameters: {num_params:,}")
    print_msg(f"vocabulary size: {tokenizer.get_vocab_size()}")

    if config['preload'] is not None:
        model_filename = get_weights_path(config, config['preload'])
        print_msg(f"Preloading model {model_filename}")

        state = torch.load(model_filename, map_location=device, weights_only=True)
        model.module.load_state_dict(state['model_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        optimiser.load_state_dict(state['optimizer_state_dict'])

        intial_epoch = state['epoch'] + 1
        global_step = state['global_step']
        
        print_msg(f"Preloaded model from epoch {intial_epoch - 1}")
        print_msg(f"Global step: {global_step}")

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(intial_epoch, config["num_epochs"]):
        model.train()

        # Set epoch in distributed sampler to guarantee unique splits/shuffles per GPU
        train_dataloader.sampler.set_epoch(epoch)

        batch_iter = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        loss_sum = 0.0

        for i, batch in enumerate(batch_iter):
            encoder_input = batch["encoder_input"].to(device, non_blocking=True)
            decoder_input = batch["decoder_input"].to(device, non_blocking=True)
            encoder_mask = batch["encoder_mask"].to(device, non_blocking=True)
            decoder_mask = batch["decoder_mask"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True) # shape (batch_size, seq_len)

            # AMP Forward Pass
            with torch.amp.autocast('cuda', dtype=torch.float16):
                # Run the tensors through the model
                proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask) # (batch_size, seq_len, vocab_size)
                loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
                raw_loss = loss.item()
            
            # AMP Backward and Optimization Step
            loss = loss / accumulation_steps
            loss_sum += raw_loss
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
                scheduler.step()
            global_step += 1

            if rank == 0:
                batch_iter.set_postfix({f"loss": f"{raw_loss:.4f}"})
                writer.add_scalar('train_loss', raw_loss, global_step)
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
        
        # Average loss across all workers for accurate evaluation
        local_loss = loss_sum / len(train_dataloader)
        loss_tensor = torch.tensor([local_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_epoch_loss = loss_tensor.item() / world_size

        if rank == 0:
            writer.flush()
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0

            if rank == 0:
                # Remove previous models
                prev_weights = glob.glob(str(Path(config["model_folder"]) / f"{config['model_basename']}*.pt"))
                for f in prev_weights:
                    if Path(f).exists():
                        os.remove(f)

                checkpoint_path = get_weights_path(config, epoch)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(), # Add this
                    "global_step": global_step
                }, checkpoint_path)


                print_msg(f"Saved checkpoint to {checkpoint_path}")
        else:
            print_msg(f"Epoch {epoch}: Loss did not improve from {best_loss}")
            patience_counter += 1


        if rank == 0:
            writer.add_scalar('epoch_loss', best_loss, epoch)
            evaluate_model(model, tokenizer, config["src_seq_len"], config["tgt_seq_len"], val_dataloader, device, lambda x: batch_iter.write(x))

        if patience_counter >= config["patience"]:
            print_msg(f"Early stopping at epoch {epoch}")
            break
    
    if rank == 0:
        if writer:
            writer.close()
    dist.destroy_process_group()
    

if __name__ == "__main__":
    config = get_config()
    train_model(config)
