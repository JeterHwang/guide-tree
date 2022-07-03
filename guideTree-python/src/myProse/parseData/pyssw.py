#!/usr/bin/env python
"""
Simple python wrapper for SSW library
Please put the path of libssw.so into LD_LIBRARY_PATH or pass it explicitly as a parameter
By Yongan Zhao (March 2016)
Revised by Mengyao Zhao on 2022-May-23
"""
import random
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO
import json
import sys
import os.path as op
import argparse as ap
import ctypes as ct
import timeit as ti
import numpy as np
import ssw_lib


def read(sFile):
    """
    read a sequence file
    @param  sFile   sequence file
    """
    def read_one_fasta(f):
        """
        read a fasta file
        @param  f   file handler
        """
        sId = ''
        sSeq = ''
        for l in f:
            if l.startswith('>'):
                if sSeq:
                    yield sId, sSeq, ''
                sId = l.strip()[1:].split()[0]
                sSeq = ''
            else:
                sSeq += l.strip()

        yield sId, sSeq, ''
# read
    with open(sFile, 'r') as f:
        for sId,sSeq,sQual in read_one_fasta(f):
            yield sId, sSeq, sQual
    
def to_int(seq, lEle, dEle2Int):
    """
    translate a sequence into numbers
    @param  seq   a sequence
    """
    num_decl = len(seq) * ct.c_int8
    num = num_decl()
    for i,ele in enumerate(seq):
        try:
            n = dEle2Int[ele]
        except KeyError:
            n = dEle2Int[lEle[-1]]
        finally:
            num[i] = n

    return num


def align_one(ssw, qProfile, rNum, nRLen, nOpen, nExt, nFlag, nMaskLen):
    """
    align one pair of sequences
    @param  qProfile   query profile
    @param  rNum   number array for reference
    @param  nRLen   length of reference sequence
    @param  nFlag   alignment flag
    @param  nMaskLen   mask length
    """
    res = ssw.ssw_align(qProfile, rNum, ct.c_int32(nRLen), nOpen, nExt, nFlag, 0, 0, nMaskLen)

    nScore = res.contents.nScore
    nScore2 = res.contents.nScore2
    nRefBeg = res.contents.nRefBeg
    nRefEnd = res.contents.nRefEnd
    nQryBeg = res.contents.nQryBeg
    nQryEnd = res.contents.nQryEnd
    nRefEnd2 = res.contents.nRefEnd2
    lCigar = [res.contents.sCigar[idx] for idx in range(res.contents.nCigarLen)]
    nCigarLen = res.contents.nCigarLen
    ssw.align_destroy(res)

    return (nScore, nScore2, nRefBeg, nRefEnd, nQryBeg, nQryEnd, nRefEnd2, nCigarLen, lCigar)


def buildPath(q, r, nQryBeg, nRefBeg, lCigar):
    """
    build cigar string and align path based on cigar array returned by ssw_align
    @param  q   query sequence
    @param  r   reference sequence
    @param  nQryBeg   begin position of query sequence
    @param  nRefBeg   begin position of reference sequence
    @param  lCigar   cigar array
    """
    sCigarInfo = 'MIDNSHP=X'
    sCigar = ''
    sQ = ''
    sA = ''
    sR = ''
    nQOff = nQryBeg
    nROff = nRefBeg
    for x in lCigar:
        n = x >> 4
        m = x & 15
        if m > 8:
            c = 'M'
        else:
            c = sCigarInfo[m]
        sCigar += str(n) + c

        if c == 'M':
            sQ += q[nQOff : nQOff+n]
            sA += ''.join(['|' if q[nQOff+j] == r[nROff+j] else '*' for j in range(n)])
            sR += r[nROff : nROff+n]
            nQOff += n
            nROff += n
        elif c == 'I':
            sQ += q[nQOff : nQOff+n]
            sA += ' ' * n
            sR += '-' * n
            nQOff += n
        elif c == 'D':
            sQ += '-' * n
            sA += ' ' * n
            sR += r[nROff : nROff+n]
            nROff += n
    return sCigar, sQ, sA, sR


def main(args):
    lEle = []
    dRc = {} 
    dEle2Int = {}
    dInt2Ele = {}
    dump_dict = []
# load AA score matrix
    if not args.sMatrix:
        lEle = 'A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   B   Z   X   *'.split()
        for i,ele in enumerate(lEle):
            dEle2Int[ele] = i
            dEle2Int[ele.lower()] = i
            dInt2Ele[i] = ele
        # nEleNum = len(lEle)
        lScore = ssw_lib.lBlosum50
    else:
        # assume the format of the input score matrix is the same as that of http://www.ncbi.nlm.nih.gov/Class/FieldGuide/BLOSUM62.txt
        lEle, dEle2Int, dInt2Ele, lScore = ssw_lib.read_matrix(args.sMatrix)

# translate score matrix to ctypes
    mat = (len(lScore) * ct.c_int8) ()
    mat[:] = lScore
# set flag
    nFlag = 0
    if args.bPath:
        nFlag = 2

    ssw = ssw_lib.CSsw(args.sLibPath)
    seqs = list(SeqIO.parse(args.input, 'fasta'))
    pairs = []
    while len(pairs) < args.train_size:
        sampled_pair = random.sample(range(len(seqs)), 2)
        pairs.append(sampled_pair) 
    data = []
# iterate query sequence
    for i, (index_A, index_B) in enumerate(tqdm(pairs)):
        id_A, seq_A = str(seqs[index_A].id), str(seqs[index_A].seq) 
        #print(id_A, seq_A)
        qNum = to_int(seq_A, lEle, dEle2Int)
        qProfile = ssw.ssw_init(qNum, ct.c_int32(len(seq_A)), mat, len(lEle), 2)
        nMaskLen = len(seq_A) // 2

        id_B, seq_B = str(seqs[index_B].id), str(seqs[index_B].seq)
        #print(id_B, seq_B)
        rNum = to_int(seq_B, lEle, dEle2Int)
        res = align_one(ssw, qProfile, rNum, len(seq_B), args.nOpen, args.nExt, nFlag, nMaskLen)
        data.append({'A' : index_A, 'B' : index_B, 'score' : res[0]})

        ssw.init_destroy(qProfile)
        
    with open(args.output_dir / 'train.json', 'w') as f:
        for dt in data:
            f.write(json.dumps(dt))
            f.write('\n')

        
if __name__ == '__main__':

    parser = ap.ArgumentParser()
    parser.add_argument('-l', '--sLibPath', default='./', help='path of libssw.so')
    parser.add_argument('-m', '--nMatch', type=int, default=2, help='a positive integer as the score for a match in genome sequence alignment. [default: 2]')
    parser.add_argument('-x', '--nMismatch', type=int, default=2, help='a positive integer as the score for a mismatch in genome sequence alignment. [default: 2]')
    parser.add_argument('-o', '--nOpen', type=int, default=3, help='a positive integer as the penalty for the gap opening in genome sequence alignment. [default: 3]')
    parser.add_argument('-e', '--nExt', type=int, default=1, help='a positive integer as the penalty for the gap extension in genome sequence alignment. [default: 1]')
    parser.add_argument('-p', '--bProtein', action='store_true', help='Do protein sequence alignment. Without this option, pyssw will do genome sequence alignment. [default: False]')
    parser.add_argument('-a', '--sMatrix', default='', help='a file for either Blosum or Pam weight matrix. [default: Blosum50]')
    parser.add_argument('-c', '--bPath', action='store_true', help='Return the alignment path. [default: False]')
    parser.add_argument('-f', '--nThr', default=0, help='a positive integer. Only output the alignments with the Smith-Waterman score >= N.')
    parser.add_argument('-r', '--bBest', action='store_true', help='The best alignment will be picked between the original read alignment and the reverse complement read alignment. [default: False]')
    parser.add_argument('-s', '--bSam', action='store_true', help='Output in SAM format. [default: no header]')
    parser.add_argument('-header', '--bHeader', action='store_true', help='If -s is used, include header in SAM output.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_size', type=int, default=1000000)
    parser.add_argument('--input', type=Path, default='./astral-scopedom-seqres-gd-all-2.08-stable.fa')
    parser.add_argument('--output_dir', type=Path, default='./')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()

    t1 = ti.default_timer()
    random.seed(args.seed)
    main(args)
    t2 = ti.default_timer()
    sys.stderr.write('CPU time: {} seconds\n'.format(t2 - t1))
