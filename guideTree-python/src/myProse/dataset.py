import torch
from alphabets import Uniprot21
from utils import read_data
from torch.nn.utils.rnn import pad_sequence

class SCOPePairsDataset:
    def __init__(
        self, 
        seq_path,
        pair_path, 
        split,
        alphabet=Uniprot21()
    ):
        print('# loading SCOP sequence pairs:')
        seqs, pairs = read_data(seq_path, pair_path)
        self.split = split
        self.seqs = [torch.from_numpy(alphabet.encode(x.encode('utf-8').upper())) for x in seqs]
        self.pairs = [(pair['A'], pair['B'], torch.tensor(pair['score'])) for pair in pairs]
        self.alphabet = alphabet
        print('# loaded', len(self.x0), 'sequence pairs')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.seqs[self.pairs[idx][0]].long(), self.seqs[self.pairs[idx][1]].long(), self.pairs[idx][2].long()
    
    def collade_fn(self, samples):
        seqA, seqB, score = samples[0], samples[1], samples[2], samples[3]
        seqA.sort(key=lambda x: len(x), reverse=True)
        seqB.sort(key=lambda x: len(x), reverse=True)
        lenA = [len(x) for x in seqA]
        lenB = [len(x) for x in seqB]
        seqA = pad_sequence(seqA, batch_first=True, padding_value=len(self.alphabet))
        seqB = pad_sequence(seqB, batch_first=True, padding_value=len(self.alphabet))
        return seqA, lenA, seqB, lenB, score