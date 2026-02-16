#!/usr/bin/env python3
"""
Simplified Markov Model for DNA Sequences
Trains a model and scores sequences from a FASTA file
"""

import argparse
import sys
from collections import defaultdict
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Markov model for DNA')
    parser.add_argument('--fasta', type=str, required=True, help='FASTA file path')
    parser.add_argument('--order', type=int, required=True, help='Model order (m)')
    return parser.parse_args()


def read_fasta_file(filename):
    """Read sequences from FASTA format"""
    seqs = []
    header = None
    seq = []
    
    try:
        f = open(filename, 'r')
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # save previous sequence
                if header is not None:
                    full_seq = ''.join(seq)
                    seqs.append((header, full_seq.upper()))
                
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        
        # don't forget last one
        if header is not None:
            full_seq = ''.join(seq)
            seqs.append((header, full_seq.upper()))
        
        f.close()
        
    except FileNotFoundError:
        print(f"Error: Can't find file {filename}", file=sys.stderr)
        sys.exit(1)
    
    return seqs


def remove_sequences_with_N(sequences):
    """Filter out any sequences that have N in them"""
    clean_seqs = []
    
    for header, seq in sequences:
        # check if N is in the sequence
        has_n = False
        for char in seq:
            if char == 'N':
                has_n = True
                break
        
        if not has_n:
            clean_seqs.append((header, seq))
    
    return clean_seqs


def count_all_kmers(sequences, k):
    """Count kmers of length k across all sequences"""
    counts = defaultdict(int)
    
    # go through each sequence
    for header, seq in sequences:
        # slide window across sequence
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            counts[kmer] += 1
    
    return counts


def build_markov_model(sequences, m):
    """
    Build markov model of order m
    For order m, we need to count (m+1)-mers and m-mers
    Then calculate probabilities
    """
    
    # count the (m+1)-mers - these are our transitions
    transition_counts = count_all_kmers(sequences, m + 1)
    
    # count the m-mers - these are our contexts
    if m > 0:
        context_counts = count_all_kmers(sequences, m)
    else:
        context_counts = None
    
    # now calculate probabilities
    # we'll store them as log probabilities
    probs = {}
    
    if m == 0:
        # order 0 is simple - just nucleotide frequencies
        total_count = 0
        for nuc in transition_counts:
            total_count += transition_counts[nuc]
        
        # calculate probability for each nucleotide
        for nuc in transition_counts:
            count = transition_counts[nuc]
            # add 1 for pseudocount (laplace smoothing)
            prob = (count + 1) / (total_count + 4)
            probs[nuc] = math.log(prob)
    
    else:
        # order m > 0
        # probability is count(transition) / count(context)
        for transition in transition_counts:
            count = transition_counts[transition]
            
            # get the context (first m characters)
            context = transition[:m]
            
            # get context count
            if context in context_counts:
                ctx_count = context_counts[context]
            else:
                ctx_count = 0
            
            # calculate probability with pseudocount
            prob = (count + 1) / (ctx_count + 4)
            probs[transition] = math.log(prob)
    
    return probs, context_counts


def score_seq(seq, m, model_probs, context_counts):
    """
    Score a sequence using the markov model
    Returns log likelihood
    """
    score = 0.0
    
    if m == 0:
        # order 0: just add up nucleotide probabilities
        for base in seq:
            if base in model_probs:
                score += model_probs[base]
            else:
                # unseen nucleotide - use small probability
                num_nucs = len(model_probs)
                score += math.log(1.0 / (num_nucs + 4))
    
    else:
        # order m: need to handle first m positions differently
        
        # for first m positions, use uniform probability
        for i in range(min(m, len(seq))):
            score += math.log(0.25)
        
        # now score the rest
        for i in range(m, len(seq)):
            # get context (previous m bases)
            context = seq[i-m:i]
            # get transition (context + current base)
            transition = seq[i-m:i+1]
            
            if transition in model_probs:
                score += model_probs[transition]
            else:
                # unseen transition - use pseudocount
                if context_counts and context in context_counts:
                    ctx_count = context_counts[context]
                else:
                    ctx_count = 0
                prob = 1.0 / (ctx_count + 4)
                score += math.log(prob)
    
    return score


def main():
    # get command line arguments
    args = parse_args()
    
    # check that order is valid
    if args.order < 0:
        print(f"Error: order must be >= 0", file=sys.stderr)
        sys.exit(1)
    
    # read the sequences from fasta file
    sequences = read_fasta_file(args.fasta)
    
    if len(sequences) == 0:
        print("Error: no sequences found", file=sys.stderr)
        sys.exit(1)
    
    # remove sequences with N
    sequences = remove_sequences_with_N(sequences)
    
    if len(sequences) == 0:
        print("Error: no valid sequences after filtering", file=sys.stderr)
        sys.exit(1)
    
    # train the model
    model_probs, context_counts = build_markov_model(sequences, args.order)
    
    # score each sequence and print
    for header, seq in sequences:
        s = score_seq(seq, args.order, model_probs, context_counts)
        print(f"{s:.6f}")


if __name__ == "__main__":
    main()
