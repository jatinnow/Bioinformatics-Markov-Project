# Markov Model for CTCF Binding Prediction (Project Milestone)

## 1. Description
This project implements a Markov Model of varying orders (0-10) to predict CTCF binding sites on Chromosome 4 using a 5-fold cross-validation approach.

## 2. Requirements
* **Language:** Python 3.x
* **Libraries:** `numpy`, `pandas`, `pysam`, `matplotlib`, `sklearn`
* **Dependencies:** Ensure `pysam` is installed to handle genomic coordinates.

## 3. Project Structure
* `MarkovCrossValidation.py`: Main script for training and k-fold evaluation.
* `simplerVersion.py`: Script for generating sequence log-likelihood scores.
* `data/`: Contains `chr4_200bp_bins.tsv`. (Note: hg38.fa is excluded due to size).
* `results/`: Contains AUROC and AUPRC plots for Order 3.

## 4. Genomic Data Setup
The human genome fasta file (`hg38.fa`) is too large for GitHub. To run the code:
1. Download the file from UCSC: [hg38.fa.gz](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz)
2. Unzip it and place it in the `data/` folder.
3. Ensure the filename matches `data/hg38.fa`.

## 5. How to Run
To execute the cross-validation for a specific order (e.g., Order 3):
```bash
python3 MarkovCrossValidation.py --file data/chr4_200bp_bins.tsv --genome data/hg38.fa --tf CTCF --order 3 --k 5
