import argparse
import random
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm

import mint.helpers.extract as extract
from mint.helpers.extract import load_config, CollateFn, MINTWrapper

extract.random = random  # MINT's extract.py is missing `import random`
random.seed(42)          # reproducible truncation of sequences > 512


class StreamingCSVDataset(IterableDataset):
    """Streams (seq1, seq2) pairs from a CSV without loading it all into memory."""
    def __init__(self, path, col1, col2, read_chunksize=100_000):
        super().__init__()
        self.path, self.col1, self.col2 = path, col1, col2
        self.read_chunksize = read_chunksize

    def __iter__(self):
        for chunk in pd.read_csv(
            self.path, usecols=[self.col1, self.col2], chunksize=self.read_chunksize
        ):
            yield from zip(chunk[self.col1], chunk[self.col2])


parser = argparse.ArgumentParser()
parser.add_argument("Inter_table", help="Protein interaction csv file (needs columns Protein1, Protein2, Seq1, Seq2)")
parser.add_argument("Output_table", help="Output csv with Protein1, Protein2 and embedding columns")
parser.add_argument("Mint_loc", help="MINT model checkpoint location")
parser.add_argument("Batch_size", type=int, help="Batch size")
parser.add_argument("device", help="cuda/cpu")
args = parser.parse_args()

device = args.device
cfg = load_config("esm2_t33_650M_UR50D.json")  # model config
checkpoint_path = args.Mint_loc

# Count rows once for a proper progress bar
n_rows = sum(
    len(c) for c in pd.read_csv(args.Inter_table, usecols=["Protein1"], chunksize=1_000_000)
)
n_batches = -(-n_rows // args.Batch_size)  # ceiling division

dataset = StreamingCSVDataset(args.Inter_table, "Seq1", "Seq2")
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.Batch_size,
    collate_fn=CollateFn(512),
    shuffle=False,
    num_workers=0,  # must stay 0 with an IterableDataset, or rows get duplicated
)

wrapper = MINTWrapper(cfg, checkpoint_path, sep_chains=True, device=device)
wrapper.eval()


csv_chunks = pd.read_csv(
    args.Inter_table, usecols=["Protein1", "Protein2"], chunksize=args.Batch_size
)

first = True
with torch.no_grad():
    for (chains, chain_ids), chunk in tqdm(zip(loader, csv_chunks), total=n_batches):
        chains = chains.to(device)
        chain_ids = chain_ids.to(device)

        emb = wrapper(chains, chain_ids)                    # (B, 2560) or (B, 2, 1280)
        emb = emb.reshape(emb.shape[0], 2, -1).mean(dim=1)  # (B, 1280): mean of the two chains
        emb = emb.cpu().numpy()

        assert len(emb) == len(chunk), "Loader and CSV chunk are out of sync"

        emb_df = pd.DataFrame(
            emb,
            index=chunk.index,
            columns=[f"Embedding_{j}" for j in range(emb.shape[1])],
        )
        id_df = pd.DataFrame(
            {"Pair": [tuple(sorted((p1, p2)))
                             for p1, p2 in zip(chunk["Protein1"], chunk["Protein2"])]},
            index=chunk.index,
        )
        pd.concat([id_df, emb_df], axis=1).to_csv(
            args.Output_table,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        first = False
