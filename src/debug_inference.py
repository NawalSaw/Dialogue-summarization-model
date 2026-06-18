import torch

from config import get_config
from train import get_model, get_dataset_tokenizer, greedy_decode

def main():
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_dataloader, tokenizer = get_dataset_tokenizer(config)

    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    print("Model loaded")

    checkpoint = torch.load("weights/tmodel_24.pt", map_location=device)
    print("Checkpoint loaded")

    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded")

    model.eval()

    for batch in test_dataloader:
        encoder_input = batch["encoder_input"].to(device)
        src_mask = batch["encoder_mask"].to(device)
        greedy_decode_output = greedy_decode(model, tokenizer, encoder_input, src_mask, config['src_seq_len'], config['tgt_seq_len'], device)
        
        print("Output:", greedy_decode_output)
        print("Label:", batch["label"])

        print("decoded_output:", tokenizer.decode(greedy_decode_output.tolist()))
        print("decoded_label:", tokenizer.decode(batch["label"][0].tolist()))
        break



if __name__ == "__main__":
    main()
