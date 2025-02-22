# description: create a long sequence as haystack, leave a question-answer pair in the haystack, ask the question in the end of the haystack.

# steps:
# 1. Create a random haystack sequence, leave 0 for mask
# 2. take the tail of the haystack as question-answer pair, 
# question is the first half of the qa, answer is the second half
# 3. insert the question-answer pair at random position along the haystack
# 4. mask the answer portion at the tail of the haystack
# 5. Return positions, masked input, target, and mask
import torch
from torch.utils.data import Dataset, DataLoader

def make_haystack(n_input_values, seq_len):
    # Create random sequence leaving 0 for mask
    seq = torch.randint(low=1, high=n_input_values, size=(1, seq_len))
    return seq

def create_qa_pair(haystack, qa_len):
    # Ensure qa_len is even
    if qa_len % 2 != 0:
        raise ValueError("qa_len must be an even number")
    # Ensure qa_len is less than half of seq_len
    if qa_len > haystack.shape[1]//2:
        raise ValueError("qa_len must be less than half of seq_len")
    # Take tail of haystack as QA pair and clone it
    qa = haystack[:, -qa_len:].clone()
    # Split into question and answer
    question = qa[:, :qa_len//2]
    answer = qa[:, qa_len//2:]
    return question, answer

def insert_qa(haystack, question, answer, qa_len):
    # Create a new tensor for the result
    result = haystack.clone()
    # Insert QA pair at random position
    insert_pos = torch.randint(low=0, high=haystack.shape[1]-qa_len*2, size=(1,)).item()
    result[:, insert_pos:insert_pos+qa_len//2] = question
    result[:, insert_pos+qa_len//2:insert_pos+qa_len] = answer

    # needle in haystack
    needle_in_haystack = torch.zeros_like(haystack)
    needle_in_haystack[:, insert_pos:insert_pos+qa_len//2] = question
    needle_in_haystack[:, insert_pos+qa_len//2:insert_pos+qa_len] = answer

    return result, insert_pos, needle_in_haystack

def mask_answer(haystack, qa_len):
    # Create copies to avoid modifying the original
    masked_haystack = haystack.clone()
    mask = torch.zeros_like(haystack)
    
    # Mask the answer at the tail of haystack
    masked_haystack[:, -qa_len//2:] = 0
    mask[:, -qa_len//2:] = 1
    
    return masked_haystack, mask

def make_record(config):
    # Create initial haystack
    haystack = make_haystack(config.n_input_values, config.seq_len)
    
    # Create QA pair
    question, answer = create_qa_pair(haystack, config.qa_len)
    
    # Insert QA pair
    haystack, insert_pos, needle_in_haystack = insert_qa(haystack, question, answer, config.qa_len)
    
    # Mask answer portion
    masked_haystack, mask = mask_answer(haystack, config.qa_len)
    
    # Create position IDs
    pos_ids = torch.arange(config.seq_len).unsqueeze(0)
    
    return {
        'pos_id': pos_ids,
        'input': masked_haystack,
        'target': haystack,
        'mask': mask,
        'needle_in_haystack': needle_in_haystack
    }

class HaystackDataset(Dataset):
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
            'needle_in_haystack': record['needle_in_haystack'].squeeze(0)
        }

def make_dataloader(config, shuffle=True):
    # Ensure qa_len is even
    if config.qa_len % 2 != 0:
        raise ValueError("qa_len must be an even number")
        
    dataset = HaystackDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers if hasattr(config, 'cpu_workers') else 0,
        pin_memory=True
    )
    return dataloader


