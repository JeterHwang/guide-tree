import torch
from alphabets import Uniprot21
from utils import read_data
from torch.nn.utils.rnn import pad_sequence

class SCOPePairsDataset:
    def __init__(
        self, 
        seq_path,
        split,
        alphabet=Uniprot21()
    ):
        print('# loading SCOP sequence pairs:')
        self.split = split
        self.seqs = read_data(seq_path)
        self.seqA = [torch.from_numpy(alphabet.encode(seq['A'].encode('utf-8').upper())) for seq in self.seqs]
        self.seqB = [torch.from_numpy(alphabet.encode(seq['B'].encode('utf-8').upper())) for seq in self.seqs]
        self.similarity = [torch.tensor(seq['score']) for seq in self.seqs]
        self.alphabet = alphabet
        print('# loaded', len(self.seqs), 'sequence pairs')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqA[idx].long(), self.seqB[idx].long(), self.similarity[idx]
    
    def collade_fn(self, samples):
        seqA = [sample[0] for sample in samples] 
        seqB = [sample[1] for sample in samples] 
        score = [sample[2] for sample in samples] 
        seqs = seqA + seqB
        lens = [len(x) for x in seqs]
        seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
        return seqs, lens, 10 * torch.tensor(score)