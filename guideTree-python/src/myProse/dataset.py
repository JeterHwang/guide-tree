import torch
from alphabets import Uniprot21
from utils import read_data

class SCOPePairsDataset:
    def __init__(
        self, 
        seq_path,
        pair_path, 
        alphabet=Uniprot21()
    ):
        print('# loading SCOP sequence pairs:')

        
        seqs, pairs = read_data(seq_path, pair_path)
        self.seqA = 
        x0 = [x.encode('utf-8').upper() for x in table['sequence_A']]
        self.x0 = [torch.from_numpy(alphabet.encode(x)) for x in x0]
        x1 = [x.encode('utf-8').upper() for x in table['sequence_B']]
        self.x1 = [torch.from_numpy(alphabet.encode(x)) for x in x1]

        self.y = torch.from_numpy(table['similarity'].values).long()

        print('# loaded', len(self.x0), 'sequence pairs')

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, i):
        return self.x0[i].long(), self.x1[i].long(), self.y[i]
    
    def collade_fn(self, samples):
