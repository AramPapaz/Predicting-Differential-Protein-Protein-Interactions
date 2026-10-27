# Predicting-Differential-Protein-Protein-Interactions
Novel machine learning model that uses protein sequence and condition-specific abundance information to predict differential interactions between two conditions. The model was trained and tested on validated context-specific PPIs from several cell-lines and showed formidable performance in predicting differential PPIs for both novel proteins and cell lines. To our knowledge, we provide the first computational model that predicts differential PPIs between two conditions.
## Installation

1. Clone the repository:
```
   git clone https://github.com/AramPapaz/Predicting-Differential-Protein-Protein-Interactions.git
   cd Predicting-Differential-Protein-Protein-Interactions
```

2. Create an environment (Python 3.11–3.13 recommended):
```
   conda create -n diffppi python=3.12
   conda activate diffppi
```

3. Install PyTorch for your hardware from https://pytorch.org/get-started/locally/
   (CPU-only or CUDA, depending on your machine).

4. Download the MINT model and note the file path where it is stored.
```
wget https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt
```
5. Download the weighted random forest model and note the file path where it is stored.
```
add link
```
6. Install the remaining dependencies:
```
pip install -r requirements.txt
```


## Usage
### 1. Prepare the input files

**MS protein intensity tables (CSV), one per condition**
- The first column holds the unique protein ID.
- The remaining columns hold the protein's abundance in each replicate.
- Abundances should be log-transformed (e.g. log2) so that each replicate column is roughly normally distributed.
- Missing values can be left empty. See `Valid_Values_Threshold` below.

**Protein sequences (FASTA), a single file for all proteins**
- Each header must be exactly the protein ID used in the MS tables.

Examples are in `example_data/`:
- `Condition1.csv`
- `Condition2.csv`
- `allproteinsSeq.txt`

### 2. Run the preprocessing & get MS based features

```
python preprocessing.py CONDITION1 CONDITION2 VALID_VALUES_THRESHOLD FASTA_FILE OUTPUT_SEQINFO OUTPUT_MSFEATURES
```

| Argument | Description |
|---|---|
| `CONDITION1` | CSV table for the first condition |
| `CONDITION2` | CSV table for the second condition |
| `VALID_VALUES_THRESHOLD` | Number between 0 and 1: the minimum fraction of replicates with a measured value for a protein to be kept (e.g. `0.7`) |
| `FASTA_FILE` | FASTA file with the protein sequences |
| `OUTPUT_SEQINFO` | Output file with the sequence information (`.csv`) |
| `OUTPUT_MSFEATURES` | Output file with the MS-derived features (`.csv`) |

Example with the provided data:

```
python preprocessing.py example_data/Condition1.csv example_data/Condition2.csv 0.7 example_data/allproteinsSeq.txt seqinfo.csv msfeatures.csv
```

### 3. Get MINT PPI embeddings

This step embeds every protein pair in the sequence information table using the pretrained [MINT](https://github.com/VarunUllanat/mint) model. Each pair is passed through MINT as two chains, and the two per-chain vectors are averaged into one 1280-dimensional embedding per interaction. The output table has one row per pair, with a `Pair` column (the sorted protein IDs) followed by the columns `Embedding__0` to `Embedding__1279`. The rows keep the order of the input table.

```
python get_embeddings.py SEQINFO_TABLE OUTPUT_EMBEDDINGS MINT_MODEL_LOCATION BATCHSIZE DEVICE
```

| Argument | Description |
|---|---|
| `SEQINFO_TABLE` | CSV produced by step 2 (`OUTPUT_SEQINFO`). It needs the columns `Protein1`, `Protein2`, `Seq1` and `Seq2`. |
| `OUTPUT_EMBEDDINGS` | Output CSV with the pair IDs and their embeddings |
| `MINT_MODEL_LOCATION` | Path to the downloaded MINT checkpoint file (see [MINT](https://github.com/VarunUllanat/mint) for the download) |
| `BATCHSIZE` | Number of protein pairs embedded at once (e.g. `8`). Larger is faster but needs more memory. |
| `DEVICE` | `cpu` or `cuda`. `cuda` needs a GPU and a CUDA build of PyTorch. |

Example:

```
python get_embeddings.py seqinfo.csv embeddings.csv /path/to/mint.ckpt 8 cpu
```

Notes:
- The table is read in chunks, so millions of interactions can be processed without loading the whole file into memory. The output file is written as the run progresses.
- Sequences longer than 512 residues are randomly truncated to 512 (a fixed seed makes this reproducible).
- On a CPU this is slow for large tables. A GPU is recommended for more than a few thousand pairs.
- `esm2_t33_650M_UR50D.json` (the model config) must be in the folder you run the script from. 

### 4. Predict differential PPIs

This step combines the MS-derived features (step 2) with the MINT embeddings (step 3) for each protein pair and scores every pair with the trained random forest model. The model file contains an ensemble of random forests, and the final score is the average of their predicted probabilities for the positive class (a differential interaction). Pairs are ranked from most to least likely.

```
python predict.py MSFEATURES_TABLE EMBEDDINGS_TABLE RF_MODEL_PATH OUTPUT_FILENAME
```

| Argument | Description |
|---|---|
| `MSFEATURES_TABLE` | CSV produced by step 2 (`OUTPUT_MSFEATURES`), with one row per pair and a `Pair` column |
| `EMBEDDINGS_TABLE` | CSV produced by step 3 (`OUTPUT_EMBEDDINGS`), with the same `Pair` column plus the embedding columns |
| `RF_MODEL_PATH` | Path to the trained model file (`.joblib`) |
| `OUTPUT_FILENAME` | Output CSV with the ranked predictions |

Example:

```
python predict.py msfeatures.csv embeddings.csv models/weighted_rf.joblib predictions.csv
```

**Output:** a CSV with two columns, sorted from highest to lowest score:
- `Pair`: the protein pair
- `Prediction`: a value between 0 and 1, the average probability across the ensemble that the pair is differentially interacting. Higher means more likely.

Notes:
- Only pairs present in **both** tables are scored. Pairs missing from either one are dropped without a warning.
- The model was trained with a specific set of feature columns, so the tables must come from steps 2 and 3 of this pipeline without changing or reordering columns.
- The model file is a pickle and requires `scikit-learn==1.9.0`. Other versions may show an `InconsistentVersionWarning` or give different results.

**How to interpret the predictions:** we recommend focusing on the ranking of the pairs rather than the exact `Prediction` values. Prioritise the pairs at the top of the output file (for example the top 100, or the top 1%) for follow-up or validation.
