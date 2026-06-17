import torch

from config import get_config
from train import get_model, get_dataset_tokenizer, greedy_decode

def main():
    config = get_config()

    _, _, test_dataloader, tokenizer = get_dataset_tokenizer(config)

    model = get_model(config, tokenizer.get_vocab_size()).to(torch.device('cpu'))
    print("Model loaded")

    checkpoint = torch.load("weights/tmodel_19.pt", map_location=torch.device('cpu'))
    print("Checkpoint loaded")

    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded")

    model.eval()

    for batch in test_dataloader:
        src = batch['src_text']
        tgt = batch['trg_text']

        src_mask = batch['encoder_mask']
        tgt_mask = batch['decoder_mask']

        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]

        encoder_output = model.encode(encoder_input, src_mask)
        decoder_output = model.decode(decoder_input, encoder_output, src_mask, tgt_mask)
        proj_output = model.project(decoder_output) # (batch_size, seq_len, vocab_size)

        print("mean:", proj_output.mean())
        print("std :", proj_output.std())
        print("max :", proj_output.max())
        print("min :", proj_output.min())

        print("proj_output:", proj_output.reshape(-1, tokenizer.get_vocab_size()))

        print("encoder_input:", encoder_input.shape)
        print("decoder_input:", decoder_input.shape)

        print("src_mask:", src_mask.shape)
        print("src_mask[0,0,0,:50]:", src_mask[0,0,0,:50])

        print("tgt_mask:", tgt_mask.shape)
        print("tgt_mask[0,0,:10,:10]:", tgt_mask[0,0,:10,:10])

        print("Encoder output shape:")
        print(encoder_output.shape)
        print("\n")
        print("Decoder output shape:")
        print(decoder_output.shape)
        print("\n")

        print("Encoder NaN:", torch.isnan(encoder_output).any())
        print("Encoder Inf:", torch.isinf(encoder_output).any())

        print("Decoder NaN:", torch.isnan(decoder_output).any())
        print("Decoder Inf:", torch.isinf(decoder_output).any())

        print("Encoder mean:", encoder_output.mean())
        print("Encoder std:", encoder_output.std())

        print("Decoder mean:", decoder_output.mean())
        print("Decoder std:", decoder_output.std())

        print("Projection shape:", proj_output.shape)

        print("Prediction IDs:")
        print(pred_ids)
        print("\n")

        print("Decoded prediction:")
        print(tokenizer.decode(pred_ids[0].tolist()))

        print("\n")
        print("SOURCE:")
        print(src[0])
        print("\n")
        print("TARGET:")
        print(tgt[0])
        print("\n")

        print("PRED:")
        print(tokenizer.decode(pred_ids[0].tolist()))

        print("BOS:", tokenizer.token_to_id("[BOS]"))
        print("EOS:", tokenizer.token_to_id("[EOS]"))
        print("PAD:", tokenizer.token_to_id("[PAD]"))
        print("UNK:", tokenizer.token_to_id("[UNK]"))

        probs = torch.softmax(proj_output[0,0], dim=-1)
        topk_probs, topk_ids = torch.topk(probs, 10)
        for p, idx in zip(topk_probs, topk_ids):
            print(
                tokenizer.id_to_token(idx.item()),
                p.item()
            )

        output = greedy_decode(
            model,
            tokenizer,
            encoder_input,
            src_mask,
            config["src_seq_len"],
            config["tgt_seq_len"],
            torch.device("cpu")
        )

        print("Output:", output)

        break



if __name__ == "__main__":
    main()
