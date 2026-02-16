#!/usr/bin/env python3
"""
simplerVersion.py - Simplified Markov Model for DNA Sequences

This script trains a Markov model of order m on DNA sequences from a FASTA file
and outputs the log-likelihood score for each sequence.

Educational Purpose: This demonstrates the core concepts of:
1. Markov models for sequence analysis
2. K-mer counting for probability estimation
3. Log-likelihood scoring

Author: JATIN RAGHUWANHSI
Course: Computational Functional Genomics, Jan sem 2026
"""

import argparse
import sys
from collections import defaultdict
import math


def parse_arguments():
    """
    Parse command-line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed arguments containing fasta path and model order
    """
    parser = argparse.ArgumentParser(
        description='Train a Markov model and score DNA sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--fasta',
        type=str,
        required=True,
        help='Path to FASTA file containing DNA sequences (e.g., data/sequences.fasta)'
    )
    
    parser.add_argument(
        '--order',
        type=int,
        required=True,
        help='Order m of the Markov model (0 for independent positions, 1+ for dependencies)'
    )
    
    return parser.parse_args()


def read_fasta(fasta_path):
    """
    Read sequences from a FASTA file.
    
    FASTA format consists of:
    - Header line starting with '>' (sequence identifier)
    - One or more lines of sequence data
    
    Args:
        fasta_path (str): Path to the FASTA file
        
    Returns:
        list: List of tuples (header, sequence) where sequences are uppercase
        
    Raises:
        FileNotFoundError: If the FASTA file doesn't exist
    """
    sequences = []
    current_header = None
    current_sequence = []
    
    try:
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Header line (starts with '>')
                if line.startswith('>'):
                    # Save previous sequence if it exists
                    if current_header is not None:
                        sequences.append((current_header, ''.join(current_sequence).upper()))
                    
                    # Start new sequence
                    current_header = line[1:]  # Remove '>' character
                    current_sequence = []
                else:
                    # Sequence line
                    current_sequence.append(line)
            
            # Don't forget the last sequence
            if current_header is not None:
                sequences.append((current_header, ''.join(current_sequence).upper()))
                
    except FileNotFoundError:
        print(f"Error: FASTA file '{fasta_path}' not found.", file=sys.stderr)
        print("Please check the file path and try again.", file=sys.stderr)
        sys.exit(1)
    
    return sequences


def filter_sequences_with_n(sequences):
    """
    Filter out sequences containing 'N' characters.
    
    According to project rules, we must remove sequences with 'N' 
    (which represent ambiguous or unknown nucleotides).
    
    Args:
        sequences (list): List of (header, sequence) tuples
        
    Returns:
        list: Filtered list containing only sequences without 'N'
    """
    filtered = []
    removed_count = 0
    
    for header, seq in sequences:
        if 'N' not in seq:
            filtered.append((header, seq))
        else:
            removed_count += 1
    
   # if removed_count > 0:
   #    print(f"# Removed {removed_count} sequence(s) containing 'N' characters", 
   #          file=sys.stderr)
    
    return filtered


def count_kmers(sequences, k):
    """
    Count all k-mers across all sequences.
    
    A k-mer is a substring of length k. For example, in "ACGT":
    - 1-mers: A, C, G, T
    - 2-mers: AC, CG, GT
    - 3-mers: ACG, CGT
    
    Args:
        sequences (list): List of (header, sequence) tuples
        k (int): Length of k-mers to count
        
    Returns:
        defaultdict: Dictionary mapping k-mer -> count
    """
    kmer_counts = defaultdict(int)
    
    for header, seq in sequences:
        # Slide a window of size k across the sequence
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_counts[kmer] += 1
    
    return kmer_counts


def train_markov_model(sequences, order):
    """
    Train a Markov model of given order on the sequences.
    
    MARKOV MODEL EXPLANATION:
    - Order 0: Each position is independent (simple nucleotide frequencies)
    - Order m: Current nucleotide depends on previous m nucleotides
    
    We model P(nucleotide | context), where context is the previous m nucleotides.
    
    IMPLEMENTATION:
    For order m, we need:
    1. Counts of (m+1)-mers: these represent transitions
       Example (m=1): "AC" means "seeing C after A"
    2. Counts of m-mers: these represent contexts
       Example (m=1): "A" is the context
    
    Then: P(C|A) = count("AC") / count("A")
    
    Args:
        sequences (list): List of (header, sequence) tuples
        order (int): Order of the Markov model
        
    Returns:
        tuple: (transition_probs, context_counts)
            - transition_probs: dict mapping (m+1)-mer -> log probability
            - context_counts: dict mapping m-mer -> count
    """
    # Count (m+1)-mers for transitions
    # Example: if order=1, we count 2-mers like "AC", "CG", etc.
    transition_counts = count_kmers(sequences, order + 1)
    
    # Count m-mers for contexts (background)
    # Example: if order=1, we count 1-mers like "A", "C", "G", "T"
    context_counts = count_kmers(sequences, order) if order > 0 else None
    
    # Calculate transition probabilities
    # We store log probabilities to avoid numerical underflow
    transition_probs = {}
    
    if order == 0:
        # Order 0: Independent positions, just nucleotide frequencies
        total = sum(transition_counts.values())
        for nucleotide, count in transition_counts.items():
            # Add pseudocount of 1 to avoid zero probabilities (Laplace smoothing)
            prob = (count + 1) / (total + 4)  # 4 possible nucleotides
            transition_probs[nucleotide] = math.log(prob)
    else:
        # Order m > 0: Conditional probabilities
        for transition, count in transition_counts.items():
            context = transition[:order]  # First m characters
            
            # P(transition | context) = count(transition) / count(context)
            # Add pseudocount of 1 for Laplace smoothing
            context_count = context_counts.get(context, 0)
            prob = (count + 1) / (context_count + 4)  # 4 possible next nucleotides
            
            transition_probs[transition] = math.log(prob)
    
    return transition_probs, context_counts


def score_sequence(sequence, order, transition_probs, context_counts):
    """
    Calculate the log-likelihood score for a sequence under the Markov model.
    
    LOG-LIKELIHOOD EXPLANATION:
    - We want to calculate P(sequence | model)
    - For a sequence s = s₁s₂...sₙ:
      * Order 0: log P(s) = Σᵢ log P(sᵢ)
      * Order m: log P(s) = Σᵢ log P(sᵢ | sᵢ₋ₘ...sᵢ₋₁)
    
    - Higher scores indicate the sequence is more likely under the model
    - Log-likelihood avoids numerical underflow from multiplying many small probabilities
    
    Args:
        sequence (str): DNA sequence to score
        order (int): Order of the Markov model
        transition_probs (dict): Log probabilities from training
        context_counts (dict): Context counts from training (for order > 0)
        
    Returns:
        float: Log-likelihood score for the sequence
    """
    log_likelihood = 0.0
    
    if order == 0:
        # Order 0: Sum log probabilities of individual nucleotides
        for nucleotide in sequence:
            if nucleotide in transition_probs:
                log_likelihood += transition_probs[nucleotide]
            else:
                # Unseen nucleotide: use small probability (pseudocount only)
                log_likelihood += math.log(1 / (sum(1 for k in transition_probs) + 4))
    
    else:
        # Order m > 0: Use context-dependent probabilities
        # Note: First m positions need special handling
        
        # For the first m positions, use lower-order models or uniform distribution
        # Here, we use a simple approximation: score them with pseudocount probability
        for i in range(min(order, len(sequence))):
            log_likelihood += math.log(0.25)  # Uniform probability for first m positions
        
        # Score the rest of the sequence using the m-th order model
        for i in range(order, len(sequence)):
            context = sequence[i-order:i]  # Previous m nucleotides
            transition = sequence[i-order:i+1]  # Context + current nucleotide
            
            if transition in transition_probs:
                log_likelihood += transition_probs[transition]
            else:
                # Unseen transition: use pseudocount probability
                context_count = context_counts.get(context, 0) if context_counts else 0
                prob = 1 / (context_count + 4)
                log_likelihood += math.log(prob)
    
    return log_likelihood



def main():
    """
    Main function: orchestrates the entire workflow.
    """
    # 1. Parse arguments
    args = parse_arguments()
    
    # Validate order
    if args.order < 0:
        # Keeping error prints is usually okay, but let's exit cleanly
        sys.exit(1)
    
    # 2. Read sequences
    sequences = read_fasta(args.fasta)
    
    if len(sequences) == 0:
        sys.exit(1)
    
    # 3. Filter sequences with 'N'
    sequences = filter_sequences_with_n(sequences)
    
    if len(sequences) == 0:
        sys.exit(1)
    
    # 4. Train the Markov model
    transition_probs, context_counts = train_markov_model(sequences, args.order)
    
    # 5. Score each sequence and print ONLY the results
    for header, seq in sequences:
        score = score_sequence(seq, args.order, transition_probs, context_counts)
        # This is the ONLY print statement that should remain
        print(f"{score:.6f}")

if __name__ == "__main__":
    main()

