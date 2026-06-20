import torch

from config import get_config
from train import get_model, get_dataset_tokenizer, greedy_decode

def main():
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_dataloader, tokenizer = get_dataset_tokenizer(config)

    # model = get_model(config, tokenizer.get_vocab_size()).to(device)
    # print("Model loaded")

    # checkpoint = torch.load("weights/tmodel_24.pt", map_location=device)
    # print("Checkpoint loaded")

    # model.load_state_dict(checkpoint['model_state_dict'])
    # print("Model loaded")

    # model.eval()

    for batch in test_dataloader:
        encoder_input = batch["encoder_input"].to(device)
        src_text = batch["src_text"]
        tgt_text = batch["tgt_text"]
        src_mask = batch["encoder_mask"].to(device)

        print("PAD:", tokenizer.token_to_id("[PAD]"))
        print("BOS:", tokenizer.token_to_id("[BOS]"))
        print("EOS:", tokenizer.token_to_id("[EOS]"))
        print("UNK:", tokenizer.token_to_id("[UNK]"))
        print("SPEAKER_1:", tokenizer.token_to_id("<SPEAKER_1>"))
        print("SPEAKER_2:", tokenizer.token_to_id("<SPEAKER_2>"))
        print("SPEAKER_3:", tokenizer.token_to_id("<SPEAKER_3>"))
        print("SPEAKER_4:", tokenizer.token_to_id("<SPEAKER_4>"))
        print("SPEAKER_5:", tokenizer.token_to_id("<SPEAKER_5>"))
        print("SPEAKER_6:", tokenizer.token_to_id("<SPEAKER_6>"))
        print("TURN:", tokenizer.token_to_id("<TURN>"))
        print("END_TURN:", tokenizer.token_to_id("</TURN>"))
        print()

        print("Source Tokens:")
        print(encoder_input[0])
        print("Target Tokens:")
        print(batch["decoder_input"][0])

        print("Source Text:")
        print(str(tokenizer.decode(encoder_input[0].tolist())))
        print(src_text[0])
        print("Target Text:")
        print(str(tokenizer.decode(batch["label"][0].tolist())))
        print(tgt_text)
            

        break



if __name__ == "__main__":
    main()
