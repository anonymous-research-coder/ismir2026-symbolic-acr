import torch
from torch.utils.data import Dataset
from GridMLM_tokenizers_old import CSGridMLMTokenizer
import os
import numpy as np
from music21 import converter, note, chord, harmony, meter, stream
import torch.nn.functional as F
from tqdm import tqdm
import pickle

def compute_normalized_token_entropy(logits, target_ids, pad_token_id=None):
    vocab_size = logits.shape[-1]
    max_entropy = torch.log2(torch.tensor(vocab_size, dtype=torch.float32)).item()
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log2(probs + 1e-09)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    if pad_token_id is not None:
        mask = (target_ids != pad_token_id).float()
        entropy = entropy * mask
        entropy_per_token = entropy.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    else:
        entropy_per_token = entropy.mean(dim=-1)
    entropy_per_batch = entropy_per_token.mean().item()
    return (entropy_per_token / max_entropy, entropy_per_batch / max_entropy)

class CSGridMLMDataset(Dataset):

    def __init__(self, root_dir, tokenizer, frontloading=True, refrontload=False, name_suffix='MLMH'):
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl') or file.endswith('.musicxml') or file.endswith('.mid') or file.endswith('.midi'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.tokenizer = tokenizer
        self.frontloading = frontloading
        if self.frontloading:
            root_dir = root_dir[:-1] if root_dir[-1] == '/' else root_dir
            frontloaded_file = root_dir + '_' + name_suffix + '.pickle'
            if refrontload or not os.path.isfile(frontloaded_file):
                print('Frontloading data.')
                self.encoded = []
                for data_file in tqdm(self.data_files):
                    try:
                        self.encoded.append(self.tokenizer.encode(data_file))
                    except Exception as e:
                        print('Problem in:', data_file)
                        print(e)
                if frontloaded_file is not None:
                    with open(frontloaded_file, 'wb') as f:
                        pickle.dump(self.encoded, f)
            else:
                print('Loading data file.')
                with open(frontloaded_file, 'rb') as f:
                    self.encoded = pickle.load(f)

    def __len__(self):
        if self.frontloading:
            return len(self.encoded)
        else:
            return len(self.data_files)

    def __getitem__(self, idx):
        if self.frontloading:
            encoded = self.encoded[idx]
        else:
            data_file = self.data_files[idx]
            encoded = self.tokenizer.encode(data_file)
        return {'harmony_ids': encoded['harmony_ids'], 'attention_mask': encoded['attention_mask'], 'pianoroll': encoded['pianoroll'], 'time_signature': encoded['time_signature'], 'h_density_complexity': encoded['h_density_complexity']}

def CSGridMLM_collate_fn(batch):
    harmony_ids = [torch.tensor(item['harmony_ids'], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
    time_signature = [torch.tensor(item['time_signature'], dtype=torch.float) for item in batch]
    h_density_complexity = [torch.tensor(item['h_density_complexity'], dtype=torch.float) for item in batch]
    pianorolls = [torch.tensor(item['pianoroll'], dtype=torch.float) for item in batch]
    return {'harmony_ids': torch.stack(harmony_ids), 'attention_mask': torch.stack(attention_mask), 'time_signature': torch.stack(time_signature), 'h_density_complexity': torch.stack(h_density_complexity), 'pianoroll': torch.stack(pianorolls)}
