"""
Markov Model Cross-Validation for TF Binding Prediction
========================================================
Complete script including MarkovModel class, cross-validation, and plotting.

Author: JATIN RAGHUWANSHI
Date: February 2026
Course: Computational Functional Genomics
Intermediate Milestone

Usage:
------
python MarkovCrossValidation.py \
    --file data/chr1_200bp_bins.tsv \
    --genome data/hg38.fa \
    --tf CTCF \
    --order 3 \
    --k 5
"""

import argparse
import pandas as pd
import numpy as np
import pysam
import time
import os
import sys
import math
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# =============================================================================
# MARKOV MODEL CLASS
# =============================================================================

class MarkovModel:
    """
    A Markov Model for DNA sequence classification.
    
    A Markov model of order m predicts the next nucleotide based on the 
    previous m nucleotides. For example:
    - Order 0: P(base) - each base is independent
    - Order 1: P(base | prev_base) - depends on 1 previous base
    - Order 2: P(base | prev_2_bases) - depends on 2 previous bases
    
    Mathematical Framework:
    ----------------------
    For a sequence S = s1, s2, ..., sn, the probability under a Markov model
    of order m is:
    
    P(S) = P(s1...sm) * ∏(i=m+1 to n) P(si | s(i-m)...s(i-1))
    
    In log space (to avoid numerical underflow):
    log P(S) = log P(s1...sm) + ∑(i=m+1 to n) log P(si | s(i-m)...s(i-1))
    
    Log-Likelihood Ratio (LLR):
    ---------------------------
    To classify a sequence, we compute:
    Score(S) = log P(S | Positive) - log P(S | Negative)
    
    If Score > 0: sequence is more likely from positive class (bound)
    If Score < 0: sequence is more likely from negative class (unbound)
    """
    
    def __init__(self, order):
        """
        Initialize the Markov Model.
        
        Parameters:
        -----------
        order : int
            The order of the Markov model (m). For order m, we consider
            m previous nucleotides when predicting the next one.
        """
        self.order = order
        
        # We'll count k-mers of length (order + 1)
        # For order m, we need to see m bases + 1 target base = (m+1)-mer
        self.kmer_length = order + 1
        
        # Dictionaries to store k-mer counts
        # Using defaultdict so missing k-mers automatically get count of 0
        self.pos_kmer_counts = defaultdict(int)
        self.neg_kmer_counts = defaultdict(int)
        
        # Dictionaries to store log probabilities after training
        self.pos_log_probs = {}
        self.neg_log_probs = {}
        
        # Flag to check if model has been trained
        self.is_trained = False
    
    
    def train(self, positive_sequences, negative_sequences, pseudocount=1.0):
        """
        Train the Markov model on positive and negative sequence sets.
        
        This method:
        1. Counts all k-mers in positive and negative sequences
        2. Adds pseudocounts to avoid zero probabilities
        3. Converts counts to log probabilities
        
        Parameters:
        -----------
        positive_sequences : list of str
            DNA sequences from bound regions (labeled as 'B')
        negative_sequences : list of str
            DNA sequences from unbound regions (labeled as 'U')
        pseudocount : float, default=1.0
            Laplace smoothing parameter. Adding pseudocount avoids zero
            probabilities for unseen k-mers. Typical value is 1.0.
            
        Mathematical Details:
        ---------------------
        Without smoothing: P(kmer) = count(kmer) / total_count
        With smoothing: P(kmer) = (count(kmer) + α) / (total_count + α * V)
        where α = pseudocount, V = vocabulary size (4^(m+1) for DNA)
        
        Why pseudocounts?
        -----------------
        If a k-mer never appears in training, its probability would be 0,
        and log(0) = -infinity. This would make scoring impossible.
        Pseudocounts ensure every k-mer has a small non-zero probability.
        """
        
        print(f"Training Markov Model of order {self.order}...")
        print(f"  K-mer length: {self.kmer_length}")
        print(f"  Positive sequences: {len(positive_sequences)}")
        print(f"  Negative sequences: {len(negative_sequences)}")
        
        # ===================================================================
        # STEP 1: Count k-mers in positive sequences
        # ===================================================================
        for seq in positive_sequences:
            # Slide a window of size kmer_length across the sequence
            # For a sequence of length n, we get (n - kmer_length + 1) k-mers
            for i in range(len(seq) - self.kmer_length + 1):
                kmer = seq[i:i + self.kmer_length]
                self.pos_kmer_counts[kmer] += 1
        
        # ===================================================================
        # STEP 2: Count k-mers in negative sequences
        # ===================================================================
        for seq in negative_sequences:
            for i in range(len(seq) - self.kmer_length + 1):
                kmer = seq[i:i + self.kmer_length]
                self.neg_kmer_counts[kmer] += 1
        
        print(f"  Unique k-mers in positive: {len(self.pos_kmer_counts)}")
        print(f"  Unique k-mers in negative: {len(self.neg_kmer_counts)}")
        
        # ===================================================================
        # STEP 3: Calculate vocabulary size for smoothing
        # ===================================================================
        # Vocabulary size V = 4^(m+1) for DNA sequences
        # This represents all possible k-mers of length (order+1)
        vocab_size = 4 ** self.kmer_length
        print(f"  Vocabulary size (4^{self.kmer_length}): {vocab_size}")
        
        # ===================================================================
        # STEP 4: Convert counts to log probabilities for positive model
        # ===================================================================
        # Total count with pseudocount smoothing:
        # Total = sum(all counts) + pseudocount * vocab_size
        pos_total = sum(self.pos_kmer_counts.values()) + pseudocount * vocab_size
        
        # For k-mers we've seen, use their actual count + pseudocount
        for kmer, count in self.pos_kmer_counts.items():
            prob = (count + pseudocount) / pos_total
            self.pos_log_probs[kmer] = math.log(prob)
        
        # For unseen k-mers, we'll compute probability on-the-fly during scoring
        # Store the default log probability for unseen k-mers
        self.pos_default_log_prob = math.log(pseudocount / pos_total)
        
        # ===================================================================
        # STEP 5: Convert counts to log probabilities for negative model
        # ===================================================================
        neg_total = sum(self.neg_kmer_counts.values()) + pseudocount * vocab_size
        
        for kmer, count in self.neg_kmer_counts.items():
            prob = (count + pseudocount) / neg_total
            self.neg_log_probs[kmer] = math.log(prob)
        
        # Default log probability for unseen k-mers in negative model
        self.neg_default_log_prob = math.log(pseudocount / neg_total)
        
        self.is_trained = True
        print(f"  Training complete!")
    
    
    def score(self, sequence):
        """
        Score a DNA sequence using the trained Markov model.
        
        Returns the Log-Likelihood Ratio (LLR):
        Score = log P(sequence | Positive) - log P(sequence | Negative)
        
        Parameters:
        -----------
        sequence : str
            DNA sequence to score (should contain only A, C, G, T)
            
        Returns:
        --------
        score : float
            Log-Likelihood Ratio. Positive values indicate higher likelihood
            of being from the positive class (bound region).
            
        Mathematical Details:
        ---------------------
        For a sequence S with k-mers k1, k2, ..., kn:
        
        log P(S | Model) = ∑(i=1 to n) log P(ki | Model)
        
        LLR = log P(S | Pos) - log P(S | Neg)
            = ∑(i=1 to n) [log P(ki | Pos) - log P(ki | Neg)]
            = ∑(i=1 to n) log[P(ki | Pos) / P(ki | Neg)]
        
        This is the sum of log odds ratios for each k-mer in the sequence.
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring!")
        
        # Check if sequence is long enough
        if len(sequence) < self.kmer_length:
            # Cannot score sequences shorter than k-mer length
            return 0.0
        
        # ===================================================================
        # STEP 1: Initialize scores
        # ===================================================================
        pos_log_prob_sum = 0.0  # Sum of log P(ki | Positive)
        neg_log_prob_sum = 0.0  # Sum of log P(ki | Negative)
        
        # ===================================================================
        # STEP 2: Slide window across sequence and accumulate log probabilities
        # ===================================================================
        for i in range(len(sequence) - self.kmer_length + 1):
            kmer = sequence[i:i + self.kmer_length]
            
            # Look up log probability for this k-mer in positive model
            # If k-mer wasn't seen during training, use default probability
            if kmer in self.pos_log_probs:
                pos_log_prob_sum += self.pos_log_probs[kmer]
            else:
                pos_log_prob_sum += self.pos_default_log_prob
            
            # Look up log probability for this k-mer in negative model
            if kmer in self.neg_log_probs:
                neg_log_prob_sum += self.neg_log_probs[kmer]
            else:
                neg_log_prob_sum += self.neg_default_log_prob
        
        # ===================================================================
        # STEP 3: Compute Log-Likelihood Ratio (LLR)
        # ===================================================================
        # LLR = log P(S | Pos) - log P(S | Neg)
        llr = pos_log_prob_sum - neg_log_prob_sum
        
        return llr


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
    --------
    args : argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='K-fold cross-validation for TF binding prediction using Markov models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python MarkovCrossValidation.py \\
        --file data/chr1_200bp_bins.tsv \\
        --genome data/hg38.fa \\
        --tf CTCF \\
        --order 3 \\
        --k 5
        """
    )
    
    parser.add_argument('--file', type=str, required=True,
                        help='Path to TSV file with TF binding data (e.g., chr1_200bp_bins.tsv)')
    
    parser.add_argument('--genome', type=str, required=True,
                        help='Path to reference genome FASTA file (e.g., hg38.fa or hg38.fa.gz)')
    
    parser.add_argument('--tf', type=str, required=True,
                        choices=['CTCF', 'REST', 'EP300'],
                        help='Transcription factor name (CTCF, REST, or EP300)')
    
    parser.add_argument('--order', type=int, required=True,
                        help='Order of the Markov model (0 to 10)')
    
    parser.add_argument('--k', type=int, required=True,
                        help='Number of folds for cross-validation (3 to 5 recommended)')
    
    parser.add_argument('--pseudocount', type=float, default=1.0,
                        help='Pseudocount for Laplace smoothing (default: 1.0)')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save output plots (default: results)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.order < 0 or args.order > 10:
        parser.error("Order must be between 0 and 10")
    
    if args.k < 2:
        parser.error("Number of folds k must be at least 2")
    
    return args


# =============================================================================
# DATA LOADING AND SEQUENCE FETCHING
# =============================================================================

def load_data(tsv_file, tf_name):
    """
    Load TF binding data from TSV file.
    
    Parameters:
    -----------
    tsv_file : str
        Path to TSV file containing binding data
    tf_name : str
        Name of the transcription factor (column name in TSV)
    
    Returns:
    --------
    df : pandas.DataFrame
        Filtered dataframe containing only rows with 'B' or 'U' labels
        for the specified TF
    
    Notes:
    ------
    The TSV file has columns: chr, start, end, ATAC, CTCF, REST, EP300
    We filter for rows where the TF column is either 'B' (bound) or 'U' (unbound)
    As per project instructions, we ignore the ATAC column for the intermediate milestone.
    """
    print(f"\n{'='*70}")
    print(f"LOADING DATA")
    print(f"{'='*70}")
    
    # Read TSV file using pandas
    # sep='\t' indicates tab-separated values
    print(f"Reading file: {tsv_file}")
    df = pd.read_csv(tsv_file, sep='\t')
    
    print(f"  Total rows in file: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Check if the TF column exists
    if tf_name not in df.columns:
        raise ValueError(f"TF '{tf_name}' not found in columns: {list(df.columns)}")
    
    # Filter for rows with 'B' or 'U' labels
    # This removes any rows with missing data or other labels
    df_filtered = df[df[tf_name].isin(['B', 'U'])].copy()
    
    print(f"  Rows with 'B' or 'U' for {tf_name}: {len(df_filtered)}")
    
    # Count class distribution
    bound_count = (df_filtered[tf_name] == 'B').sum()
    unbound_count = (df_filtered[tf_name] == 'U').sum()
    
    print(f"  Bound ('B') regions: {bound_count}")
    print(f"  Unbound ('U') regions: {unbound_count}")
    print(f"  Class ratio (B:U): 1:{unbound_count/bound_count:.2f}")
    
    if len(df_filtered) == 0:
        raise ValueError(f"No valid data found for TF '{tf_name}'")
    
    return df_filtered


def fetch_sequences(df, genome_file):
    """
    Fetch DNA sequences from genome FASTA file for each region in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with columns: chr, start, end
    genome_file : str
        Path to genome FASTA file (can be .fa or .fa.gz)
    
    Returns:
    --------
    sequences : list of str
        List of DNA sequences (uppercase) corresponding to each row
    valid_indices : list of int
        List of indices of rows with valid sequences (no 'N's)
    
    Notes:
    ------
    - Uses pysam.FastaFile for efficient random access to FASTA
    - Sequences are converted to uppercase (A, C, G, T)
    - Sequences containing 'N' are filtered out (as per project requirements)
    - The genome file must be indexed (.fai file in same directory)
    
    Important about coordinates:
    ----------------------------
    BED/TSV files use 0-based, half-open intervals [start, end)
    pysam also uses 0-based, half-open intervals
    So we can use the coordinates directly without adjustment
    """
    print(f"\n{'='*70}")
    print(f"FETCHING SEQUENCES FROM GENOME")
    print(f"{'='*70}")
    
    print(f"Opening genome file: {genome_file}")
    
    # Open the FASTA file using pysam
    # pysam requires an index file (.fai) for random access
    # If the index doesn't exist, you can create it with: samtools faidx genome.fa
    try:
        fasta = pysam.FastaFile(genome_file)
    except Exception as e:
        print(f"\nERROR: Could not open genome file.")
        print(f"Make sure the file exists and is indexed (.fai file).")
        print(f"To index: samtools faidx {genome_file}")
        raise e
    
    sequences = []
    valid_indices = []
    sequences_with_n = 0
    
    print(f"Fetching {len(df)} sequences...")
    
    # Iterate through each row in the dataframe
    for idx, row in df.iterrows():
        chrom = row['chr']
        start = int(row['start'])
        end = int(row['end'])
        
        # Fetch the sequence using pysam
        # pysam.fetch(chromosome, start, end) returns sequence for [start, end)
        try:
            seq = fasta.fetch(chrom, start, end)
        except KeyError:
            # Chromosome not found in genome file
            print(f"  WARNING: Chromosome '{chrom}' not found in genome file. Skipping.")
            continue
        except Exception as e:
            print(f"  WARNING: Error fetching {chrom}:{start}-{end}. Skipping. Error: {e}")
            continue
        
        # Convert to uppercase
        # The reference genome might have lowercase letters for repeat regions
        seq = seq.upper()
        
        # Filter out sequences containing 'N' (ambiguous bases)
        # As per project description: "Repeat regions and those with missing 
        # sequence information have been removed from both the training and 
        # prediction data"
        if 'N' in seq:
            sequences_with_n += 1
            continue
        
        # Also check for unexpected characters (should only be A, C, G, T)
        if not all(base in 'ACGT' for base in seq):
            print(f"  WARNING: Unexpected characters in sequence at {chrom}:{start}-{end}. Skipping.")
            continue
        
        sequences.append(seq)
        valid_indices.append(idx)
    
    fasta.close()
    
    print(f"  Successfully fetched: {len(sequences)} sequences")
    print(f"  Filtered out (contain 'N'): {sequences_with_n}")
    print(f"  Sequence length: {len(sequences[0]) if sequences else 0} bp (should be ~200 bp)")
    
    if len(sequences) == 0:
        raise ValueError("No valid sequences found! Check your genome file and coordinates.")
    
    return sequences, valid_indices


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_fold_curves(fpr, tpr, precision, recall, roc_auc, pr_auc, fold_idx, order, output_dir, tf_name):
    """
    Plot ROC and Precision-Recall curves for a single fold.
    Only called for fold 0 to avoid generating too many files.
    
    Parameters:
    -----------
    fpr : numpy.array
        False positive rates for ROC curve
    tpr : numpy.array
        True positive rates for ROC curve
    precision : numpy.array
        Precision values for PR curve
    recall : numpy.array
        Recall values for PR curve
    roc_auc : float
        Area under ROC curve
    pr_auc : float
        Area under PR curve
    fold_idx : int
        Fold index (should be 0)
    order : int
        Markov model order
    output_dir : str
        Directory to save plots
    tf_name : str
        Transcription factor name
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # -------------------------------------------------------------------
    # SUBPLOT 1: ROC Curve
    # -------------------------------------------------------------------
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.set_title(f'ROC Curve - {tf_name} - Order {order} - Fold {fold_idx}', 
                  fontsize=16, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------
    # SUBPLOT 2: Precision-Recall Curve
    # -------------------------------------------------------------------
    ax2.plot(recall, precision, color='darkgreen', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=14)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.set_title(f'Precision-Recall Curve - {tf_name} - Order {order} - Fold {fold_idx}', 
                  fontsize=16, fontweight='bold')
    ax2.legend(loc="best", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the combined plot
    output_file = os.path.join(output_dir, f'plot_order_{order}_fold_{fold_idx}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Plot saved to: {output_file}")
    plt.close()


def plot_all_folds_summary(all_roc_curves, all_pr_curves, avg_auroc, avg_auprc, 
                           output_dir, tf_name, order):
    """
    Plot ROC and PR curves for all folds on the same plot (summary visualization).
    
    Parameters:
    -----------
    all_roc_curves : list of dict
        List of ROC curve data for each fold
    all_pr_curves : list of dict
        List of PR curve data for each fold
    avg_auroc : float
        Average AUROC across all folds
    avg_auprc : float
        Average AUPRC across all folds
    output_dir : str
        Directory to save plots
    tf_name : str
        Transcription factor name
    order : int
        Markov model order
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # -------------------------------------------------------------------
    # SUBPLOT 1: All ROC Curves
    # -------------------------------------------------------------------
    for curve in all_roc_curves:
        ax1.plot(curve['fpr'], curve['tpr'], 
                 label=f"Fold {curve['fold']} (AUC = {curve['auc']:.3f})",
                 alpha=0.7, linewidth=2)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.set_title(f'ROC Curves - {tf_name} - Order {order}\nAverage AUC = {avg_auroc:.3f}', 
                  fontsize=16, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------
    # SUBPLOT 2: All PR Curves
    # -------------------------------------------------------------------
    for curve in all_pr_curves:
        ax2.plot(curve['recall'], curve['precision'], 
                 label=f"Fold {curve['fold']} (AUC = {curve['auc']:.3f})",
                 alpha=0.7, linewidth=2)
    
    ax2.set_xlabel('Recall', fontsize=14)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.set_title(f'Precision-Recall Curves - {tf_name} - Order {order}\nAverage AUC = {avg_auprc:.3f}', 
                  fontsize=16, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the summary plot
    output_file = os.path.join(output_dir, f'summary_all_folds_order_{order}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Summary plot saved to: {output_file}")
    plt.close()


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def perform_cross_validation(sequences, labels, order, k_folds, pseudocount, output_dir, tf_name):
    """
    Perform k-fold cross-validation using Markov models.
    
    Parameters:
    -----------
    sequences : list of str
        DNA sequences
    labels : numpy.array
        Binary labels (1 for bound 'B', 0 for unbound 'U')
    order : int
        Order of the Markov model
    k_folds : int
        Number of folds for cross-validation
    pseudocount : float
        Pseudocount for Laplace smoothing
    output_dir : str
        Directory to save output plots
    tf_name : str
        Name of the transcription factor (for plot titles)
    
    Returns:
    --------
    results : dict
        Dictionary containing AUROC and AUPRC for each fold and averages
    
    Cross-Validation Strategy:
    --------------------------
    1. Split data into k folds
    2. For each fold i:
       a. Use fold i as test set
       b. Use remaining k-1 folds as training set
       c. Split training set into positive (B) and negative (U) sequences
       d. Train Markov model on positive and negative sequences
       e. Score all sequences in test set
       f. Calculate AUROC and AUPRC
    3. Report average metrics across all folds
    """
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION")
    print(f"{'='*70}")
    
    print(f"Markov model order: {order}")
    print(f"Number of folds: {k_folds}")
    print(f"Total sequences: {len(sequences)}")
    print(f"Positive (bound): {np.sum(labels)}")
    print(f"Negative (unbound): {len(labels) - np.sum(labels)}")
    
    # Convert to numpy arrays for easier indexing
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # Initialize k-fold cross-validation
    # shuffle=True ensures random splitting
    # random_state ensures reproducibility
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_aurocs = []
    fold_auprcs = []
    
    # Store ROC and PR curves for plotting
    all_roc_curves = []
    all_pr_curves = []
    
    # ===================================================================
    # ITERATE THROUGH EACH FOLD
    # ===================================================================
    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(sequences)):
        print(f"\n{'-'*70}")
        print(f"FOLD {fold_idx}/{k_folds-1}")
        print(f"{'-'*70}")
        
        # -------------------------------------------------------------------
        # STEP 1: Split data into training and test sets
        # -------------------------------------------------------------------
        train_sequences = sequences[train_indices]
        train_labels = labels[train_indices]
        test_sequences = sequences[test_indices]
        test_labels = labels[test_indices]
        
        print(f"Training set size: {len(train_sequences)}")
        print(f"  Positive: {np.sum(train_labels)}")
        print(f"  Negative: {len(train_labels) - np.sum(train_labels)}")
        print(f"Test set size: {len(test_sequences)}")
        print(f"  Positive: {np.sum(test_labels)}")
        print(f"  Negative: {len(test_labels) - np.sum(test_labels)}")
        
        # -------------------------------------------------------------------
        # STEP 2: Separate training sequences by class
        # -------------------------------------------------------------------
        # Positive sequences: those labeled as 'B' (bound) with label=1
        positive_sequences = train_sequences[train_labels == 1].tolist()
        # Negative sequences: those labeled as 'U' (unbound) with label=0
        negative_sequences = train_sequences[train_labels == 0].tolist()
        
        if len(positive_sequences) == 0 or len(negative_sequences) == 0:
            print(f"  WARNING: Fold {fold_idx} has no positive or negative examples. Skipping.")
            continue
        
        # -------------------------------------------------------------------
        # STEP 3: Train Markov model
        # -------------------------------------------------------------------
        print(f"\nTraining Markov model...")
        model = MarkovModel(order=order)
        model.train(positive_sequences, negative_sequences, pseudocount=pseudocount)
        
        # -------------------------------------------------------------------
        # STEP 4: Score test sequences
        # -------------------------------------------------------------------
        print(f"\nScoring test sequences...")
        test_scores = []
        for seq in test_sequences:
            score = model.score(seq)
            test_scores.append(score)
        
        test_scores = np.array(test_scores)
        
        print(f"  Score statistics:")
        print(f"    Min: {np.min(test_scores):.4f}")
        print(f"    Max: {np.max(test_scores):.4f}")
        print(f"    Mean: {np.mean(test_scores):.4f}")
        print(f"    Std: {np.std(test_scores):.4f}")
        
        # -------------------------------------------------------------------
        # STEP 5: Calculate ROC curve and AUROC
        # -------------------------------------------------------------------
        # ROC curve plots True Positive Rate vs False Positive Rate
        # at different threshold values
        fpr, tpr, roc_thresholds = roc_curve(test_labels, test_scores)
        roc_auc = auc(fpr, tpr)
        
        print(f"\n  AUROC (Area Under ROC Curve): {roc_auc:.4f}")
        
        # Store for later plotting
        all_roc_curves.append({
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'fold': fold_idx
        })
        fold_aurocs.append(roc_auc)
        
        # -------------------------------------------------------------------
        # STEP 6: Calculate Precision-Recall curve and AUPRC
        # -------------------------------------------------------------------
        # PR curve is often more informative for imbalanced datasets
        # Precision = TP / (TP + FP)
        # Recall = TP / (TP + FN) = True Positive Rate
        precision, recall, pr_thresholds = precision_recall_curve(test_labels, test_scores)
        pr_auc = average_precision_score(test_labels, test_scores)
        
        print(f"  AUPRC (Area Under PR Curve): {pr_auc:.4f}")
        
        # Store for later plotting
        all_pr_curves.append({
            'precision': precision,
            'recall': recall,
            'auc': pr_auc,
            'fold': fold_idx
        })
        fold_auprcs.append(pr_auc)
        
        # -------------------------------------------------------------------
        # STEP 7: Plot curves for fold 0 only (to avoid too many files)
        # -------------------------------------------------------------------
        if fold_idx == 0:
            print(f"\n  Generating plot for fold {fold_idx}...")
            plot_fold_curves(fpr, tpr, precision, recall, roc_auc, pr_auc, 
                           fold_idx, order, output_dir, tf_name)
    
    # ===================================================================
    # COMPUTE AVERAGE METRICS
    # ===================================================================
    avg_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    avg_auprc = np.mean(fold_auprcs)
    std_auprc = np.std(fold_auprcs)
    
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Average AUROC: {avg_auroc:.4f} ± {std_auroc:.4f}")
    print(f"Average AUPRC: {avg_auprc:.4f} ± {std_auprc:.4f}")
    print(f"\nIndividual fold results:")
    for i, (auroc, auprc) in enumerate(zip(fold_aurocs, fold_auprcs)):
        print(f"  Fold {i}: AUROC = {auroc:.4f}, AUPRC = {auprc:.4f}")
    
    # ===================================================================
    # PLOT SUMMARY OF ALL FOLDS
    # ===================================================================
    print(f"\nGenerating summary plot with all folds...")
    plot_all_folds_summary(all_roc_curves, all_pr_curves, avg_auroc, avg_auprc, 
                          output_dir, tf_name, order)
    
    # Return results
    results = {
        'fold_aurocs': fold_aurocs,
        'fold_auprcs': fold_auprcs,
        'avg_auroc': avg_auroc,
        'std_auroc': std_auroc,
        'avg_auprc': avg_auprc,
        'std_auprc': std_auprc
    }
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function.
    
    Workflow:
    ---------
    1. Parse command-line arguments
    2. Load TSV data and filter for specific TF
    3. Fetch DNA sequences from genome
    4. Perform k-fold cross-validation
    5. Report results
    """
    # Record start time
    start_time = time.time()
    
    print("="*70)
    print("MARKOV MODEL CROSS-VALIDATION FOR TF BINDING PREDICTION")
    print("="*70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ===================================================================
    # STEP 1: Parse arguments
    # ===================================================================
    args = parse_arguments()
    
    print(f"\nParameters:")
    print(f"  TSV file: {args.file}")
    print(f"  Genome file: {args.genome}")
    print(f"  Transcription factor: {args.tf}")
    print(f"  Markov model order: {args.order}")
    print(f"  Number of folds: {args.k}")
    print(f"  Pseudocount: {args.pseudocount}")
    print(f"  Output directory: {args.output_dir}")
    
    # ===================================================================
    # STEP 2: Load and filter data
    # ===================================================================
    df = load_data(args.file, args.tf)
    
    # ===================================================================
    # STEP 3: Fetch sequences from genome
    # ===================================================================
    sequences, valid_indices = fetch_sequences(df, args.genome)
    
    # Filter dataframe to keep only rows with valid sequences
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    
    # ===================================================================
    # STEP 4: Prepare labels
    # ===================================================================
    # Convert 'B' to 1 (positive class) and 'U' to 0 (negative class)
    labels = (df_valid[args.tf] == 'B').astype(int).values
    
    print(f"\nFinal dataset:")
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Positive (bound): {np.sum(labels)}")
    print(f"  Negative (unbound): {len(labels) - np.sum(labels)}")
    
    # ===================================================================
    # STEP 5: Perform cross-validation
    # ===================================================================
    results = perform_cross_validation(
        sequences=sequences,
        labels=labels,
        order=args.order,
        k_folds=args.k,
        pseudocount=args.pseudocount,
        output_dir=args.output_dir,
        tf_name=args.tf
    )
    
    # ===================================================================
    # STEP 6: Report final results
    # ===================================================================
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"EXECUTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"\nFinal Results:")
    print(f"  Average AUROC: {results['avg_auroc']:.4f} ± {results['std_auroc']:.4f}")
    print(f"  Average AUPRC: {results['avg_auprc']:.4f} ± {results['std_auprc']:.4f}")
    print(f"\nPlots saved to: {args.output_dir}/")
    print(f"  - plot_order_{args.order}_fold_0.png (individual fold 0)")
    print(f"  - summary_all_folds_order_{args.order}.png (all folds)")
    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Entry point for the script.
    This allows the script to be run from command line.
    """
    try:
        main()
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR OCCURRED")
        print(f"{'='*70}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"\nFor help, run: python MarkovCrossValidation.py --help")
        print(f"{'='*70}")
        sys.exit(1)