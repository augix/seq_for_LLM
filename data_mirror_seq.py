import torch
from torch.utils.data import Dataset, DataLoader
# require arguments:
# config.n_input_values, config.seq_len, config.mask_frac, config.batch_size

# --------------------------------
#  Create data for testing
# --------------------------------
def make_seq(n_input_values, seq_len):
    seq = torch.randint(low=1, high=n_input_values, size=(1, seq_len)) # leave 0 for mask
    # make 2nd half of sequence equal to 1st half
    half_len = seq_len // 2
    seq[:, half_len:] = seq[:, :half_len].flip(1)
    return seq

def masking(seq, mask_frac):
    seq_len = seq.size(1)
    mask = torch.rand(size=(1, seq_len)) < mask_frac
    masked_seq = torch.where(mask, torch.zeros_like(seq), seq)
    return masked_seq, mask

def make_record(config):
    seq = make_seq(config.n_input_values, config.seq_len)
    ids = torch.arange(config.seq_len).reshape(1, -1)
    masked_seq, mask = masking(seq, config.mask_frac)
    return {'pos_id': ids, 'input': masked_seq, 'target': seq, 'mask': mask}

def make_batch(config):
    records = [make_record(config) for _ in range(config.batch_size)]
    batch = {
        'pos_id': torch.cat([record['pos_id'] for record in records]),
        'input': torch.cat([record['input'] for record in records]),
        'target': torch.cat([record['target'] for record in records]),
        'mask': torch.cat([record['mask'] for record in records])}
    return batch

class MirrorSeqDataset(Dataset):
    def __init__(self, config):
        self.config = config
    
    def __len__(self):
        return self.config.num_samples
    
    def __getitem__(self, idx):
        record = make_record(self.config)
        return {
            'pos_id': record['pos_id'].squeeze(0),
            'input': record['input'].squeeze(0),
            'target': record['target'].squeeze(0),
            'mask': record['mask'].squeeze(0),
        }

def make_dataloader(config, shuffle=True):
    if config.seq_len % 2 != 0:
        raise ValueError("seq_len must be an even number")
        
    dataset = MirrorSeqDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers if hasattr(config, 'cpu_workers') else 0,
        pin_memory=True
    )
    return dataloader