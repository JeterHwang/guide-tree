# -*- coding: utf8 -*-
"""
needle.py -- Needleman Wunsch pairwise alignment function

Revision:
    4-April-2014
"""

import cProfile
import sys
import numpy #use numpy matrices  to score and backtrack matrices

numpy.set_printoptions(linewidth=200)
numpy.set_printoptions(edgeitems=10)

def needle(s1, s2, score_fun, verbose=False, semi=False, debug=False):
    """ (str, str, fun(), ...) -> (str, str)
    Optimally align two sequences according to scoring formula supplied

    Args:
        s1, s2 (str): sequences to be aligned
        score_fun: function to score a.a.-a.a. and a.a.-gap alignments
        verbose: flag for printing alignment to screen
        semi: semi-global alignment: no terminal gap penalties
        debug: flag for printing debugging information - matrices etc.
    Raises:
        None
    Returns:
        tuple (str, str) containing aligned sequences
    """
    #internally, have s1 as the longer sequence:
    #(fixes bug in edge cases for semi-global alignment)
    s1 = s1
    s2 = s2
    if len(s1) < len(s2):
        (s1, s2) = (s2, s1)
        swapped = True
        if debug:
            print("(Note: swapping s1 and s2 during alignment)") 
    else:
        swapped = False
    
    #Mostly want to be case insensitive when aligning. 
    #Converting both sequences to upper case here is more efficient 
    #than doing that in the score function many times, without much
    #loss of generality
    (original_s1, original_s2) = (s1, s2)#keep case sensitive copies
    s1 = s1.upper()
    s2 = s2.upper()
    
    #matrix dimensions, with extra row/column for leading gaps:
    N = len(s1) + 1  #columns
    M = len(s2) + 1  #rows
    
    score_matrix = numpy.float_([[0]*N]*M)
    #backtrack matrix, stores 0 for left, 1 for diagonal, 2 for up:
    back_matrix = numpy.float_([[0]*N]*M)
    
    #fill in first row, col of both matrices:
    for col in range(1, N):
        if semi is True:
            gap = 0
        else:
            gap = score_fun(s1[col-1], '-')
        score_matrix[0][col] = score_matrix[0][col-1] + gap
        back_matrix[0][col] = 0
    
    for row in range(1, M):
        if semi is True:
            gap = 0
        else:
            gap = score_fun(s2[row-1], '-')
        score_matrix[row][0] = score_matrix[row-1][0] + gap
        back_matrix[row][0] = 2
    
    #fill in rest of score_matrix and back_matrix:
    for row in range(1, M):
        for col in range(1, N):
            s = score_fun(s1[col-1], s2[row-1])
            lgap = score_fun(s1[col-1], '-')
            ugap = score_fun('-', s2[row-1])
            (l, ul, u) = (score_matrix[row][col-1], score_matrix[row-1][col-1],
                        score_matrix[row-1][col])
            if max(l+lgap, ul+s, u+ugap) == l + lgap:
                score_matrix[row][col] = l + lgap
                back_matrix[row][col] = 0
            elif max(l+lgap, ul+s, u+ugap) == u + ugap:
                score_matrix[row][col] = u + ugap
                back_matrix[row][col] = 2
            else:
                score_matrix[row][col] = ul + s
                back_matrix[row][col] = 1
    if debug:
        print(score_matrix)
        print(back_matrix)
    
    #Backtrack:
    #restore case sensitive versions of sequences for backtracking:
    (s1, s2) = (original_s1, original_s2)
    #Create sequences as a list, then join()
    #Create lists backwards (appending to end instead of beginning),i
    #then reverse when we're done
    aligned1 = []
    aligned2 = []
    (row, col) = (M-1, N-1)
    start_col = col #start backtracking from this column
    if semi:
        max_val = max(score_matrix[row, 0:N])
        start_col = [c for c in range(N) if score_matrix[row][c] == max_val][-1]
    while col > start_col:
        aligned2.append('-')
        aligned1.append(s1[col-1])
        col -= 1
    while row or col:
        if back_matrix[row][col] == 0:
            aligned2.append('-')
            aligned1.append(s1[col-1])
            col -= 1
        elif back_matrix[row][col] == 1:
            aligned1.append(s1[col-1])
            aligned2.append(s2[row-1])
            col -= 1
            row -= 1
        else: #back_matrix[row][col] == 2:
            aligned1.append('-')
            aligned2.append(s2[row-1])
            row -= 1
    aligned1.reverse()
    aligned2.reverse()
    aligned1 = ''.join(aligned1)
    aligned2 = ''.join(aligned2)
    
    #if we swapped the input strings earlier, swap them back now
    if swapped:
        (aligned1, aligned2) = (aligned2, aligned1)
    
    if verbose:
        pretty_print(aligned1, aligned2)
        
    return (aligned1, aligned2)

#This dictionary is global to avoid the overhead of initialising it inside every
#call of the function blosum62
BLOSUM62 = {'*': {'*': 1, 'A': -7, 'C': -7, 'B': -7, 'E': -7, 'D': -7, 'G': -7,
                  'F': -7, 'I': -7, 'H': -7, 'K': -7, 'M': -7, 'L': -7, 'N': -7,
                  'Q': -7, 'P': -7, 'S': -7, 'R': -7, 'T': -7, 'W': -7, 'V': -7,
                  'Y': -7, 'X': -7, 'Z': -7}, 
            'A': {'*': -7, 'A': 6, 'C': -2, 
                  'B': -3, 'E': -2, 'D': -3, 'G': -1, 'F': -4, 'I': -3, 'H': -3,
                  'K': -2, 'M': -2, 'L': -3, 'N': -2, 'Q': -1, 'P': -1, 'S': 1, 
                  'R': -2, 'T': -1, 'W': -4, 'V': -1, 'Y': -4, 'X': -1, 
                  'Z': -2},
            'C': {'*': -7, 'A': -2, 'C': 9, 'B': -5, 'E': -7, 'D': -6, 'G': -5,
                  'F': -3, 'I': -2, 'H': -6, 'K': -5, 'M': -3, 'L': -3, 'N': -4,
                  'Q': -5, 'P': -5, 'S': -2, 'R': -6, 'T': -2, 'W': -5, 'V': -2,
                  'Y': -4, 'X': -4, 'Z': -6},
            'B': {'*': -7, 'A': -3, 'C': -5, 'B': 4, 'E': 0, 'D': 4, 'G': -2,
                  'F': -5, 'I': -5, 'H': -1, 'K': -1, 'M': -5, 'L': -5, 'N': 4,
                  'Q': -1, 'P': -4, 'S': -1, 'R': -2, 'T': -2, 'W': -6, 
                  'V': -5, 'Y': -4, 'X': -2, 'Z': 1}, 
            'E': {'*': -7, 'A': -2, 'C': -7, 'B': 0, 'E': 6, 'D': 1, 'G': -4,
                  'F': -5, 'I': -5, 'H': -1, 'K': 0, 'M': -4, 'L': -5, 'N': -1,
                  'Q': 1, 'P': -3, 'S': -1, 'R': -2, 'T': -2, 'W': -5, 'V': -4, 
                  'Y': -4, 'X': -2, 'Z': 5}, 
            'D': {'*': -7, 'A': -3, 'C': -6, 'B': 4, 'E': 1, 'D': 7, 'G': -3,
                  'F': -5, 'I': -6, 'H': -2, 'K': -2, 'M': -5, 'L': -6, 'N': 1,
                  'Q': -2, 'P': -3, 'S': -2, 'R': -3, 'T': -2, 'W': -7, 
                  'V': -5, 'Y': -5, 'X': -3, 'Z': 0}, 
            'G': {'*': -7, 'A': -1, 'C': -5, 'B': -2, 'E': -4, 'D': -3, 'G': 6,
                  'F': -5, 'I': -6, 'H': -4, 'K': -3, 'M': -5, 'L': -6,
                  'N': -2, 'Q': -4, 'P': -4, 'S': -1, 'R': -4, 'T': -3, 'W': -5,
                  'V': -5, 'Y': -6, 'X': -3, 'Z': -4}, 
            'F': {'*': -7, 'A': -4, 'C': -3, 'B': -5, 'E': -5, 'D': -5, 'G': -5,
                  'F': 7, 'I': -1, 'H': -3, 'K': -4, 'M': -1, 'L': 0, 'N': -5, 
                  'Q': -4, 'P': -5, 'S': -4, 'R': -4, 'T': -3, 'W': 0, 'V': -2, 
                  'Y': 3, 'X': -3, 'Z': -5}, 
            'I': {'*': -7, 'A': -3, 'C': -2, 'B': -5, 'E': -5, 'D': -6, 'G': -6, 
                  'F': -1, 'I': 6, 'H': -5, 'K': -4, 'M': 1, 'L': 1, 'N': -5, 
                  'Q': -4, 'P': -5, 'S': -4, 'R': -5, 'T': -2, 'W': -4, 'V': 2, 
                  'Y': -3, 'X': -2, 'Z': -5}, 
            'H': {'*': -7, 'A': -3, 'C': -6, 'B': -1, 'E': -1, 'D': -2, 'G': -4,
                  'F': -3, 'I': -5, 'H': 9, 'K': -2, 'M': -3, 'L': -4, 'N': 0, 
                  'Q': 0, 'P': -4, 'S': -2, 'R': -1, 'T': -3, 'W': -4, 'V': -5, 
                  'Y': 1, 'X': -3, 'Z': -1}, 
            'K': {'*': -7, 'A': -2, 'C': -5, 'B': -1, 'E': 0, 'D': -2, 'G': -3, 
                  'F': -4, 'I': -4, 'H': -2, 'K': 6, 'M': -2, 'L': -4, 'N': -1, 
                  'Q': 1, 'P': -2, 'S': -1, 'R': 2, 'T': -2, 'W': -6, 'V': -4, 
                  'Y': -4, 'X': -2, 'Z': 0}, 
            'M': {'*': -7, 'A': -2, 'C': -3, 'B': -5, 'E': -4, 'D': -5, 'G': -5,
                  'F': -1, 'I': 1, 'H': -3, 'K': -2, 'M': 8, 'L': 2, 'N': -4, 
                  'Q': -1, 'P': -4, 'S': -3, 'R': -3, 'T': -2, 'W': -2, 'V': 0,
                  'Y': -3, 'X': -2, 'Z': -3}, 
            'L': {'*': -7, 'A': -3, 'C': -3, 'B': -5, 'E': -5, 'D': -6, 'G': -6,
                  'F': 0, 'I': 1, 'H': -4, 'K': -4, 'M': 2, 'L': 5, 'N': -5, 
                  'Q': -3, 'P': -5, 'S': -4, 'R': -4, 'T': -3, 'W': -4, 'V': 0,
                  'Y': -3, 'X': -2, 'Z': -4}, 
            'N': {'*': -7, 'A': -2, 'C': -4, 'B': 4, 'E': -1, 'D': 1, 'G': -2, 
                  'F': -5, 'I': -5, 'H': 0, 'K': -1, 'M': -4, 'L': -5, 'N': 7, 
                  'Q': -1, 'P': -4, 'S': 0, 'R': -1, 'T': -1, 'W': -6, 'V': -4,
                  'Y': -4, 'X': -2, 'Z': -1}, 
            'Q': {'*': -7, 'A': -1, 'C': -5, 'B': -1, 'E': 1, 'D': -2, 'G': -4, 
                  'F': -4, 'I': -4, 'H': 0, 'K': 1, 'M': -1, 'L': -3, 'N': -1, 
                  'Q': 7, 'P': -2, 'S': -1, 'R': 0, 'T': -2, 'W': -4, 'V': -4, 
                  'Y': -3, 'X': -2, 'Z': 4}, 
            'P': {'*': -7, 'A': -1, 'C': -5, 'B': -4, 'E': -3, 'D': -3, 'G': -4, 
                  'F': -5, 'I': -5, 'H': -4, 'K': -2, 'M': -4, 'L': -5, 'N': -4, 
                  'Q': -2, 'P': 8, 'S': -2, 'R': -3, 'T': -3, 'W': -5, 'V': -4, 
                  'Y': -5, 'X': -3, 'Z': -3}, 
            'S': {'*': -7, 'A': 1, 'C': -2, 'B': -1, 'E': -1, 'D': -2, 'G': -1, 
                  'F': -4, 'I': -4, 'H': -2, 'K': -1, 'M': -3, 'L': -4, 'N': 0, 
                  'Q': -1, 'P': -2, 'S': 6, 'R': -2, 'T': 1, 'W': -4, 'V': -3, 
                  'Y': -3, 'X': -1, 'Z': -1}, 
            'R': {'*': -7, 'A': -2, 'C': -6, 'B': -2, 'E': -2, 'D': -3, 'G': -4,
                  'F': -4, 'I': -5, 'H': -1, 'K': 2, 'M': -3, 'L': -4, 'N': -1, 
                  'Q': 0, 'P': -3, 'S': -2, 'R': 7, 'T': -2, 'W': -5, 'V': -4, 
                  'Y': -4, 'X': -2, 'Z': -1}, 
            'T': {'*': -7, 'A': -1, 'C': -2, 'B': -2, 'E': -2, 'D': -2, 'G': -3, 
                  'F': -3, 'I': -2, 'H': -3, 'K': -2, 'M': -2, 'L': -3, 'N': -1, 
                  'Q': -2, 'P': -3, 'S': 1, 'R': -2, 'T': 6, 'W': -5, 'V': -1, 
                  'Y': -3, 'X': -1, 'Z': -2}, 
            'W': {'*': -7, 'A': -4, 'C': -5, 'B': -6, 'E': -5, 'D': -7, 'G': -5, 
                  'F': 0, 'I': -4, 'H': -4, 'K': -6, 'M': -2, 'L': -4, 'N': -6, 
                  'Q': -4, 'P': -5, 'S': -4, 'R': -5, 'T': -5, 'W': 11, 'V': -3, 
                  'Y': 1, 'X': -4, 'Z': -4}, 
            'V': {'*': -7, 'A': -1, 'C': -2, 'B': -5, 'E': -4, 'D': -5, 'G': -5, 
                  'F': -2, 'I': 2, 'H': -5, 'K': -4, 'M': 0, 'L': 0, 'N': -4, 
                  'Q': -4, 'P': -4, 'S': -3, 'R': -4, 'T': -1, 'W': -3, 'V': 5,
                  'Y': -3, 'X': -2, 'Z': -4}, 
            'Y': {'*': -7, 'A': -4, 'C': -4, 'B': -4, 'E': -4, 'D': -5, 'G': -6,
                  'F': 3, 'I': -3, 'H': 1, 'K': -4, 'M': -3, 'L': -3, 'N': -4,
                  'Q': -3, 'P': -5, 'S': -3, 'R': -4, 'T': -3, 'W': 1, 'V': -3, 
                  'Y': 8, 'X': -3, 'Z': -4}, 
            'X': {'*': -7, 'A': -1, 'C': -4, 'B': -2, 'E': -2, 'D': -3, 'G': -3,
                  'F': -3, 'I': -2, 'H': -3, 'K': -2, 'M': -2, 'L': -2, 'N': -2,
                  'Q': -2, 'P': -3, 'S': -1, 'R': -2, 'T': -1, 'W': -4, 'V': -2,
                  'Y': -3, 'X': -2, 'Z': -2}, 
            'Z': {'*': -7, 'A': -2, 'C': -6, 'B': 1, 'E': 5, 'D': 0, 'G': -4, 
                  'F': -5, 'I': -5, 'H': -1, 'K': 0, 'M': -3, 'L': -4, 'N': -1, 
                  'Q': 4, 'P': -3, 'S': -1, 'R': -1, 'T': -2, 'W': -4, 'V': -4, 
                  'Y': -4, 'X': -2, 'Z': 4}}

def blosum62(aa1, aa2):
    """(str, str) -> int
    Score alignment with BLOSUM62 matrix
        
    Args:
        aa1, aa2 (str): character representing amino acid or gap '-'
    Returns:
        score of aligment of aa1 and aa2 (int)
    Raises:
        KeyError if aa1 or aa2 are not valid single letter amino acid codes
        or gaps
    """
    if aa1 == '-' or aa1 == '.':
        aa1 = '*'
    if aa2 == '-' or aa2 == '-':
        aa2 = '*'
    try:
        return BLOSUM62[aa1][aa2]
    except KeyError:
        raise KeyError("non A.A. character in sequence")


def pretty_print(aligned1, aligned2, width=60):
    """ (str, str, ...) -> None
    Pretty print alignment of two sequences
    
    Args:
        aligned1, aligned2 -- strings of equal length representing
         two aligned proteins
        width -- width of rows to print
    Returns: 
        Nothing
    Raises:
        Exception if aligned1 and aligned2 are unequal lengths
    """
    if len(aligned1) != len(aligned2):
        raise Exception("sequences must be aligned (same length)")
    to_print = len(aligned1)
    printed = 0
    while to_print > 0:
        if to_print < width:
            cols = to_print
        else:
            cols = width
        print(aligned1[printed:printed+cols])
        for i in range(cols):
            if aligned1[printed+i] == aligned2[printed+i]:
                sys.stdout.write("|")
            else:
                sys.stdout.write(" ")
        sys.stdout.write("\n")
        print(aligned2[printed:printed+cols])
        to_print -= cols
        printed += cols
        print("")


if __name__ == "__main__":
    s1 = ("SNADVTPLSLGIETLGGIMTKLITRNTTIPTKKSQVFSTAADGQTQVQIKVFQGEREMATSNKLLG"
          "QFSLVGIPPAPRGVPQVEVTFDIDANGIVNVSARDRGTGKEQQIVIQSSGGLSKDQIENMIKEAEK"
          "NAAEDAKRKELVEVINQAE")
    s2 = ("VIGIDLGTTNSCVSIMEGKTPKVIENAEGVRTTPSTVAFTADGERLVGAPAKRQAVTNSANTLFAT"
          "KRLIGRRYEDPEVQKDLKVVPYKIVKASNGDAWVEAQGKVYSPSQVGAFVLMKMKETAESYLGTTV"
          "NNAVVTVPAYFNDSQRQATKDAGQISGLNVLRVINEPTAAALAYGLDKDAGDKIIAVYDLGGGTFD"
          "VSILEIQKGVFEVKSTNGDTFLGGEDFDHALVHHLVGEFKKEQGVDLTKDPQAMQRLREAAEKAKC"
          "ELSSTTQTDINLPYITMDQSGPKHLNLKLTRAKFEQIVGDLIKRTIEPCRKALHDAEVKSSQIADV"
          "LLVGGMSRMPKVQATVQEIFGKVPSKAVNPDEAVAMGAAIQGAVLAGDVTDVLLLDVTPLSLGIET"
          "LGGIMTKLITRNTTIPTKKSQVFSTAADGQTQVQIKVFQGEREMATSNKLLGQFSLVGIPPAPRGV"
          "PQVEVTFDIDANGIVNVSARDRGTGKEQQIVIQSSGGLSKDQIENMIKEAEKNAAEDAKRKELVEV"
          "INQAEGIIHDTEAKMTEFADQLPKDECEALRTKIADTKKILDNKDNETPEAIKEACNTLQQQSLKL"
          "FEAAYK")
    cProfile.run('needle(s1,s2,BLOSUM62, semi=True, debug=False, verbose=True)')        
