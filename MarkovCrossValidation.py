"""
Markov Model Cross-Validation for TF Binding Prediction
Course Project - Computational Functional Genomics
Author: [Jatin Raghuwanshi]
Date: February 2026

This script does k-fold cross-validation to predict TF binding using Markov models.

Usage:
python MarkovCrossValidation.py --file data/chr1_200bp_bins.tsv --genome data/hg38.fa --tf CTCF --order 3 --k 5
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
    Markov Model for DNA sequences.
    
    Order 0 = independent bases
    Order 1 = depends on 1 previous base
    Order 2 = depends on 2 previous bases, etc.
    
    We score sequences by computing: log P(seq | positive) - log P(seq | negative)
    """
    
    def __init__(self, order):
        self.order = order
        self.kmer_len = order + 1  # for order m, we need (m+1)-mers
        
        # dictionaries to store kmer counts
        self.pos_counts = defaultdict(int)
        self.neg_counts = defaultdict(int)
        
        # store log probabilities after training
        self.pos_probs = {}
        self.neg_probs = {}
        
        self.trained = False
    
    
    def train(self, pos_seqs, neg_seqs, pseudocount=1.0):
        """
        Train the model on positive and negative sequences.
        
        pos_seqs = list of sequences from bound regions
        neg_seqs = list of sequences from unbound regions
        pseudocount = smoothing parameter (default 1.0)
        """
        
        print(f"Training model with order {self.order}...")
        print(f"  Kmer length: {self.kmer_len}")
        print(f"  Positive sequences: {len(pos_seqs)}")
        print(f"  Negative sequences: {len(neg_seqs)}")
        
        # count all kmers in positive sequences
        for seq in pos_seqs:
            # slide a window across the sequence
            for i in range(len(seq) - self.kmer_len + 1):
                kmer = seq[i:i + self.kmer_len]
                self.pos_counts[kmer] += 1
        
        # count all kmers in negative sequences
        for seq in neg_seqs:
            for i in range(len(seq) - self.kmer_len + 1):
                kmer = seq[i:i + self.kmer_len]
                self.neg_counts[kmer] += 1
        
        print(f"  Found {len(self.pos_counts)} unique kmers in positive")
        print(f"  Found {len(self.neg_counts)} unique kmers in negative")
        
        # calculate vocab size for smoothing
        # vocab size = 4^(order+1) for DNA
        vocab_size = 4 ** self.kmer_len
        print(f"  Total possible kmers: {vocab_size}")
        
        # convert counts to log probabilities with pseudocounts
        # formula: P(kmer) = (count + pseudocount) / (total + pseudocount * vocab_size)
        
        # positive model
        pos_total = 0
        for count in self.pos_counts.values():
            pos_total += count
        pos_total += pseudocount * vocab_size
        
        for kmer in self.pos_counts:
            prob = (self.pos_counts[kmer] + pseudocount) / pos_total
            self.pos_probs[kmer] = math.log(prob)
        
        # default probability for unseen kmers
        self.pos_default = math.log(pseudocount / pos_total)
        
        # negative model
        neg_total = 0
        for count in self.neg_counts.values():
            neg_total += count
        neg_total += pseudocount * vocab_size
        
        for kmer in self.neg_counts:
            prob = (self.neg_counts[kmer] + pseudocount) / neg_total
            self.neg_probs[kmer] = math.log(prob)
        
        # default probability for unseen kmers
        self.neg_default = math.log(pseudocount / neg_total)
        
        self.trained = True
        print("  Training done!")
    
    
    def score(self, seq):
        """
        Score a sequence using log-likelihood ratio.
        
        Returns: log P(seq | positive) - log P(seq | negative)
        
        Positive score = more likely to be bound
        Negative score = more likely to be unbound
        """
        
        if not self.trained:
            raise ValueError("Need to train the model first!")
        
        # sequences shorter than kmer length can't be scored
        if len(seq) < self.kmer_len:
            return 0.0
        
        # accumulate log probabilities
        pos_score = 0.0
        neg_score = 0.0
        
        # slide window and sum up log probs for each kmer
        for i in range(len(seq) - self.kmer_len + 1):
            kmer = seq[i:i + self.kmer_len]
            
            # look up positive probability
            if kmer in self.pos_probs:
                pos_score += self.pos_probs[kmer]
            else:
                pos_score += self.pos_default
            
            # look up negative probability
            if kmer in self.neg_probs:
                neg_score += self.neg_probs[kmer]
            else:
                neg_score += self.neg_default
        
        # return the log likelihood ratio
        return pos_score - neg_score


# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='K-fold CV for TF binding prediction')
    
    parser.add_argument('--file', type=str, required=True,
                        help='TSV file with binding data')
    
    parser.add_argument('--genome', type=str, required=True,
                        help='Genome FASTA file (hg38.fa)')
    
    parser.add_argument('--tf', type=str, required=True,
                        choices=['CTCF', 'REST', 'EP300'],
                        help='Which TF to predict')
    
    parser.add_argument('--order', type=int, required=True,
                        help='Markov model order (0-10)')
    
    parser.add_argument('--k', type=int, required=True,
                        help='Number of folds for CV')
    
    parser.add_argument('--pseudocount', type=float, default=1.0,
                        help='Pseudocount for smoothing (default=1.0)')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Where to save plots')
    
    args = parser.parse_args()
    
    # check that parameters are valid
    if args.order < 0 or args.order > 10:
        parser.error("Order must be 0-10")
    
    if args.k < 2:
        parser.error("Need at least 2 folds")
    
    return args


# =============================================================================
# LOAD DATA FROM TSV FILE
# =============================================================================

def load_data(tsv_file, tf_name):
    """
    Load the TSV file and filter for rows with B or U labels for the TF.
    
    Returns a dataframe with only valid rows.
    """
    print(f"\n{'='*70}")
    print(f"LOADING DATA")
    print(f"{'='*70}")
    
    print(f"Reading: {tsv_file}")
    df = pd.read_csv(tsv_file, sep='\t')
    
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # make sure TF column exists
    if tf_name not in df.columns:
        raise ValueError(f"Can't find {tf_name} in the file!")
    
    # keep only rows with B or U labels
    df = df[df[tf_name].isin(['B', 'U'])].copy()
    
    print(f"  Rows with B or U for {tf_name}: {len(df)}")
    
    # count how many bound vs unbound
    bound = (df[tf_name] == 'B').sum()
    unbound = (df[tf_name] == 'U').sum()
    
    print(f"  Bound regions: {bound}")
    print(f"  Unbound regions: {unbound}")
    print(f"  Ratio: 1:{unbound/bound:.2f}")
    
    if len(df) == 0:
        raise ValueError(f"No data found for {tf_name}!")
    
    return df


# =============================================================================
# FETCH SEQUENCES FROM GENOME
# =============================================================================

def fetch_sequences(df, genome_file):
    """
    Get DNA sequences from the genome file for each row in the dataframe.
    
    Returns:
    - list of sequences (uppercase, no N's)
    - list of valid indices (which rows had good sequences)
    """
    print(f"\n{'='*70}")
    print(f"FETCHING SEQUENCES")
    print(f"{'='*70}")
    
    print(f"Opening genome: {genome_file}")
    
    # open the FASTA file with pysam
    try:
        fasta = pysam.FastaFile(genome_file)
    except Exception as e:
        print(f"\nERROR: Can't open genome file!")
        print(f"Make sure it's indexed with: samtools faidx {genome_file}")
        raise e
    
    sequences = []
    valid_idx = []
    n_count = 0
    
    print(f"Fetching {len(df)} sequences...")
    
    # go through each row
    for idx, row in df.iterrows():
        chrom = row['chr']
        start = int(row['start'])
        end = int(row['end'])
        
        # get the sequence
        try:
            seq = fasta.fetch(chrom, start, end)
        except KeyError:
            print(f"  WARNING: Can't find chromosome {chrom}")
            continue
        except Exception as e:
            print(f"  WARNING: Error at {chrom}:{start}-{end}")
            continue
        
        # convert to uppercase
        seq = seq.upper()
        
        # skip sequences with N (ambiguous bases)
        if 'N' in seq:
            n_count += 1
            continue
        
        # make sure it's only ACGT
        valid = True
        for base in seq:
            if base not in 'ACGT':
                valid = False
                break
        
        if not valid:
            print(f"  WARNING: Bad characters at {chrom}:{start}-{end}")
            continue
        
        sequences.append(seq)
        valid_idx.append(idx)
    
    fasta.close()
    
    print(f"  Got {len(sequences)} valid sequences")
    print(f"  Filtered out {n_count} sequences with N")
    if len(sequences) > 0:
        print(f"  Sequence length: {len(sequences[0])} bp")
    
    if len(sequences) == 0:
        raise ValueError("No valid sequences found!")
    
    return sequences, valid_idx


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_fold_curves(fpr, tpr, precision, recall, roc_auc, pr_auc, 
                     fold_num, order, output_dir, tf_name):
    """
    Plot ROC and PR curves for a single fold.
    Only called for fold 0 to avoid too many files.
    """
    
    # make output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.set_title(f'ROC - {tf_name} - Order {order} - Fold {fold_num}', 
                  fontsize=16, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall curve
    ax2.plot(recall, precision, color='darkgreen', lw=2, 
             label=f'PR (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=14)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.set_title(f'Precision-Recall - {tf_name} - Order {order} - Fold {fold_num}', 
                  fontsize=16, fontweight='bold')
    ax2.legend(loc="best", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # save the plot
    filename = os.path.join(output_dir, f'plot_order_{order}_fold_{fold_num}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n  Saved plot: {filename}")
    plt.close()


def plot_all_folds(roc_data, pr_data, avg_roc, avg_pr, output_dir, tf_name, order):
    """
    Plot all folds together on one figure.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # plot all ROC curves
    for data in roc_data:
        ax1.plot(data['fpr'], data['tpr'], 
                 label=f"Fold {data['fold']} (AUC = {data['auc']:.3f})",
                 alpha=0.7, linewidth=2)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.set_title(f'ROC Curves - {tf_name} - Order {order}\nAverage AUC = {avg_roc:.3f}', 
                  fontsize=16, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # plot all PR curves
    for data in pr_data:
        ax2.plot(data['recall'], data['precision'], 
                 label=f"Fold {data['fold']} (AUC = {data['auc']:.3f})",
                 alpha=0.7, linewidth=2)
    
    ax2.set_xlabel('Recall', fontsize=14)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.set_title(f'PR Curves - {tf_name} - Order {order}\nAverage AUC = {avg_pr:.3f}', 
                  fontsize=16, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f'summary_all_folds_order_{order}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n  Saved summary plot: {filename}")
    plt.close()


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cross_validation(sequences, labels, order, k_folds, pseudocount, 
                         output_dir, tf_name):
    """
    Do k-fold cross-validation.
    
    For each fold:
    1. Split into train and test
    2. Train model on train set
    3. Score test set
    4. Calculate metrics
    """
    
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION")
    print(f"{'='*70}")
    
    print(f"Order: {order}")
    print(f"Folds: {k_folds}")
    print(f"Total sequences: {len(sequences)}")
    print(f"Positive: {np.sum(labels)}")
    print(f"Negative: {len(labels) - np.sum(labels)}")
    
    # convert to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # set up k-fold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # store results
    auroc_list = []
    auprc_list = []
    
    roc_data = []
    pr_data = []
    
    # loop through folds
    fold_num = 0
    for train_idx, test_idx in kf.split(sequences):
        print(f"\n{'-'*70}")
        print(f"FOLD {fold_num}/{k_folds-1}")
        print(f"{'-'*70}")
        
        # split the data
        train_seqs = sequences[train_idx]
        train_labels = labels[train_idx]
        test_seqs = sequences[test_idx]
        test_labels = labels[test_idx]
        
        print(f"Train size: {len(train_seqs)}")
        print(f"  Positive: {np.sum(train_labels)}")
        print(f"  Negative: {len(train_labels) - np.sum(train_labels)}")
        print(f"Test size: {len(test_seqs)}")
        print(f"  Positive: {np.sum(test_labels)}")
        print(f"  Negative: {len(test_labels) - np.sum(test_labels)}")
        
        # separate positive and negative training sequences
        pos_seqs = []
        neg_seqs = []
        for i in range(len(train_seqs)):
            if train_labels[i] == 1:
                pos_seqs.append(train_seqs[i])
            else:
                neg_seqs.append(train_seqs[i])
        
        if len(pos_seqs) == 0 or len(neg_seqs) == 0:
            print(f"  WARNING: No positive or negative examples in fold {fold_num}")
            fold_num += 1
            continue
        
        # train the model
        print(f"\nTraining model...")
        model = MarkovModel(order=order)
        model.train(pos_seqs, neg_seqs, pseudocount=pseudocount)
        
        # score test sequences
        print(f"\nScoring test sequences...")
        scores = []
        for seq in test_seqs:
            score = model.score(seq)
            scores.append(score)
        
        scores = np.array(scores)
        
        print(f"  Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        print(f"  Mean: {np.mean(scores):.4f}")
        print(f"  Std: {np.std(scores):.4f}")
        
        # calculate ROC curve
        fpr, tpr, _ = roc_curve(test_labels, scores)
        roc_auc = auc(fpr, tpr)
        
        print(f"\n  AUROC: {roc_auc:.4f}")
        
        roc_data.append({
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'fold': fold_num
        })
        auroc_list.append(roc_auc)
        
        # calculate PR curve
        precision, recall, _ = precision_recall_curve(test_labels, scores)
        pr_auc = average_precision_score(test_labels, scores)
        
        print(f"  AUPRC: {pr_auc:.4f}")
        
        pr_data.append({
            'precision': precision,
            'recall': recall,
            'auc': pr_auc,
            'fold': fold_num
        })
        auprc_list.append(pr_auc)
        
        # plot fold 0 only
        if fold_num == 0:
            print(f"\n  Making plot for fold {fold_num}...")
            plot_fold_curves(fpr, tpr, precision, recall, roc_auc, pr_auc, 
                           fold_num, order, output_dir, tf_name)
        
        fold_num += 1
    
    # calculate averages
    avg_auroc = np.mean(auroc_list)
    std_auroc = np.std(auroc_list)
    avg_auprc = np.mean(auprc_list)
    std_auprc = np.std(auprc_list)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Average AUROC: {avg_auroc:.4f} ± {std_auroc:.4f}")
    print(f"Average AUPRC: {avg_auprc:.4f} ± {std_auprc:.4f}")
    print(f"\nPer-fold:")
    for i in range(len(auroc_list)):
        print(f"  Fold {i}: AUROC={auroc_list[i]:.4f}, AUPRC={auprc_list[i]:.4f}")
    
    # make summary plot
    print(f"\nMaking summary plot...")
    plot_all_folds(roc_data, pr_data, avg_auroc, avg_auprc, 
                   output_dir, tf_name, order)
    
    results = {
        'auroc_list': auroc_list,
        'auprc_list': auprc_list,
        'avg_auroc': avg_auroc,
        'std_auroc': std_auroc,
        'avg_auprc': avg_auprc,
        'std_auprc': std_auprc
    }
    
    return results


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function - runs the whole pipeline.
    """
    
    start_time = time.time()
    
    print("="*70)
    print("MARKOV MODEL CV FOR TF BINDING")
    print("="*70)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # parse arguments
    args = parse_arguments()
    
    print(f"\nSettings:")
    print(f"  File: {args.file}")
    print(f"  Genome: {args.genome}")
    print(f"  TF: {args.tf}")
    print(f"  Order: {args.order}")
    print(f"  Folds: {args.k}")
    print(f"  Pseudocount: {args.pseudocount}")
    print(f"  Output: {args.output_dir}")
    
    # load data
    df = load_data(args.file, args.tf)
    
    # get sequences
    sequences, valid_idx = fetch_sequences(df, args.genome)
    
    # keep only valid rows
    df = df.iloc[valid_idx].reset_index(drop=True)
    
    # make labels (B=1, U=0)
    labels = []
    for i in range(len(df)):
        if df[args.tf].iloc[i] == 'B':
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)
    
    print(f"\nFinal dataset:")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Positive: {np.sum(labels)}")
    print(f"  Negative: {len(labels) - np.sum(labels)}")
    
    # run cross-validation
    results = run_cross_validation(
        sequences=sequences,
        labels=labels,
        order=args.order,
        k_folds=args.k,
        pseudocount=args.pseudocount,
        output_dir=args.output_dir,
        tf_name=args.tf
    )
    
    # print final results
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"DONE!")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"\nFinal scores:")
    print(f"  AUROC: {results['avg_auroc']:.4f} ± {results['std_auroc']:.4f}")
    print(f"  AUPRC: {results['avg_auprc']:.4f} ± {results['std_auprc']:.4f}")
    print(f"\nPlots in: {args.output_dir}/")
    print(f"  - plot_order_{args.order}_fold_0.png")
    print(f"  - summary_all_folds_order_{args.order}.png")
    print(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


# =============================================================================
# RUN IT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR!")
        print(f"{'='*70}")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print(f"\nRun with --help for usage info")
        print(f"{'='*70}")
        sys.exit(1)
