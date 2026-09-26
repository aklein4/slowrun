"""Download a bounded, revision-pinned FineWeb slice for full-model smoke tests."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
import torch
from huggingface_hub import HfApi, HfFileSystem


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, default=524544)
    parser.add_argument("--val-tokens", type=int, default=16392)
    parser.add_argument("--revision")
    args = parser.parse_args()
    if min(args.train_tokens, args.val_tokens) < 2049:
        parser.error("Each split needs at least one full 2049-token sequence")
    args.output.mkdir(parents=True, exist_ok=False)
    repo = "HuggingFaceFW/fineweb"
    revision = args.revision or HfApi().dataset_info(repo).sha
    files = sorted(
        path
        for path in HfApi().list_repo_files(
            repo, repo_type="dataset", revision=revision
        )
        if path.startswith("sample/10BT/") and path.endswith(".parquet")
    )
    filesystem = HfFileSystem()

    def documents():
        for filename in files:
            with filesystem.open(
                f"datasets/{repo}@{revision}/{filename}", "rb"
            ) as stream:
                parquet = pq.ParquetFile(stream)
                # Synchronous column reads avoid Arrow scanner workers surviving
                # partial iterator consumption and crashing interpreter shutdown.
                for batch in parquet.iter_batches(
                    batch_size=128, columns=["text"], use_threads=False
                ):
                    yield from batch.column(0).to_pylist()

    dataset_iter = documents()
    encoder = tiktoken.get_encoding("gpt2")
    manifest = {
        "dataset": repo,
        "revision": revision,
        "config": "sample-10BT",
        "tokenizer": "tiktoken:gpt2",
        "split_order": ["val", "train"],
        "files": {},
    }
    for split, budget in [("val", args.val_tokens), ("train", args.train_tokens)]:
        tokens, document_count = [], 0
        while len(tokens) < budget:
            document = next(dataset_iter)
            tokens.extend(
                ([50256] + encoder.encode_ordinary(document))[: budget - len(tokens)]
            )
            document_count += 1
        # The next split starts at a new document; a truncated tail is discarded.
        path = args.output / f"{split}.pt"
        torch.save({"tokens": torch.tensor(tokens, dtype=torch.int32)}, path)
        manifest["files"][split] = {
            "tokens": len(tokens),
            "documents": document_count,
            "sha256": file_sha256(path),
        }
    dataset_iter.close()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
