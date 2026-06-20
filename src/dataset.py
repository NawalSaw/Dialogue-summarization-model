import torch
from torch.utils.data import Dataset

class SamsumDataset(Dataset):
    def __init__(self, dataset, tokenizer, src_seq_len, tgt_seq_len):
        super().__init__()

        self.dataset = dataset 
        self.tokenizer = tokenizer
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len

        self.pad_token = tokenizer.token_to_id("[PAD]")
        self.bos_token = torch.tensor([tokenizer.token_to_id("[BOS]")], dtype=torch.int16)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int16)

    def __len__(self):
        return len(self.dataset)

    def replace_speakers(self, dialogue):
        lines = dialogue.split("\n")

        speaker_map = {}
        processed = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if ":" not in line:
                processed.append(line)
                continue

            speaker, utterance = line.split(":", 1)

            speaker = speaker.strip()
            utterance = utterance.strip()

            if speaker not in speaker_map:
                speaker_map[speaker] = len(speaker_map) + 1

            processed.append(
                f"<SPEAKER_{speaker_map[speaker]}> <TURN>{utterance}</TURN>"
            )

        return "\n".join(processed)

    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = self.replace_speakers(src_target_pair["dialogue"])
        tgt_text = src_target_pair["summary"]
        
        src_encoded = self.tokenizer.encode(src_text)
        tgt_encoded = self.tokenizer.encode(tgt_text)
        
        enc_num_padding_tokens = self.src_seq_len - len(src_encoded.ids) - 2
        dec_num_padding_tokens = self.tgt_seq_len - len(tgt_encoded.ids) - 1
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sequence length is too short")
        
        # shape (seq_len)
        encoder_input = torch.cat([
            self.bos_token,
            torch.tensor(src_encoded.ids, dtype=torch.int16),
            self.eos_token,
            torch.full(
                (enc_num_padding_tokens,),
                self.pad_token,
                dtype=torch.int16
            )
        ])
        
        # shape (seq_len)   
        decoder_input = torch.cat([
            self.bos_token,
            torch.tensor(tgt_encoded.ids, dtype=torch.int16),
            torch.full(
                (dec_num_padding_tokens,),
                self.pad_token,
                dtype=torch.int16
            )
        ])

        label = torch.cat([
            torch.tensor(tgt_encoded.ids, dtype=torch.int16),
            self.eos_token, 
            torch.full(
                (dec_num_padding_tokens,),
                self.pad_token,
                dtype=torch.int16
            )
        ])

        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0)
        decoder_mask = (
            (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0)
            & causal_mask(decoder_input.size(0))
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def get_speaker_index(speaker_id):
    return f"<SPEAKER_{speaker_id}>"