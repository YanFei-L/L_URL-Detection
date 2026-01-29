import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def _default_paths(base_dir: str) -> Dict[str, str]:
    return {
        "white_list": os.path.join(base_dir, "data", "white_list.csv"),
        "block_list": os.path.join(base_dir, "data", "block_list.csv"),
        "output": os.path.join(base_dir, "round2_dl_baseline_metrics.json"),
    }


class _CharVocab:
    def __init__(self, chars: List[str]):
        self.pad_id = 0
        self.unk_id = 1
        self._id_to_char: List[str] = ["<pad>", "<unk>"] + list(chars)
        self._char_to_id: Dict[str, int] = {c: i for i, c in enumerate(self._id_to_char)}

    @property
    def size(self) -> int:
        return len(self._id_to_char)

    def encode(self, s: str) -> List[int]:
        return [self._char_to_id.get(ch, self.unk_id) for ch in s]


def _build_vocab(urls: List[str], max_vocab_size: Optional[int]) -> _CharVocab:
    counts: Dict[str, int] = {}
    for u in urls:
        for ch in u:
            counts[ch] = counts.get(ch, 0) + 1

    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if max_vocab_size is not None:
        items = items[: int(max_vocab_size)]

    chars = [c for c, _ in items]
    return _CharVocab(chars)


def _pad_or_truncate(ids: List[int], max_len: int, pad_id: int) -> List[int]:
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


def _torch_import():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        from torch.utils.data import DataLoader, Dataset  # noqa: F401

        return True
    except Exception:
        return False


def _train_eval_torch(
    model_type: str,
    train_urls: List[str],
    train_labels: np.ndarray,
    test_urls: List[str],
    test_labels: np.ndarray,
    vocab: _CharVocab,
    max_len: int,
    embed_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    threshold: float,
) -> Dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    class URLDataset(Dataset):
        def __init__(self, urls: List[str], labels: np.ndarray):
            self.urls = urls
            self.labels = labels.astype(np.float32)

        def __len__(self) -> int:
            return int(len(self.urls))

        def __getitem__(self, idx: int):
            s = self.urls[idx]
            ids = vocab.encode(s)
            ids = _pad_or_truncate(ids, max_len=max_len, pad_id=vocab.pad_id)
            x = torch.tensor(ids, dtype=torch.long)
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y

    class CharCNN(nn.Module):
        def __init__(self, vocab_size: int, emb_dim: int):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=vocab.pad_id)
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(emb_dim, 64, kernel_size=3, padding=1),
                    nn.Conv1d(emb_dim, 64, kernel_size=4, padding=2),
                    nn.Conv1d(emb_dim, 64, kernel_size=5, padding=2),
                ]
            )
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(64 * 3, 1)

        def forward(self, x):
            e = self.emb(x)
            e = e.transpose(1, 2)
            pooled = []
            for conv in self.convs:
                z = torch.relu(conv(e))
                z = torch.max(z, dim=2).values
                pooled.append(z)
            h = torch.cat(pooled, dim=1)
            h = self.dropout(h)
            return self.fc(h).squeeze(1)

    class CharBiLSTM(nn.Module):
        def __init__(self, vocab_size: int, emb_dim: int, hidden: int = 96):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=vocab.pad_id)
            self.lstm = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden * 2, 1)

        def forward(self, x):
            e = self.emb(x)
            out, (h_n, _) = self.lstm(e)
            h = torch.cat([h_n[0], h_n[1]], dim=1)
            h = self.dropout(h)
            return self.fc(h).squeeze(1)

    if model_type == "char_cnn":
        model = CharCNN(vocab_size=vocab.size, emb_dim=embed_dim)
    elif model_type == "bilstm":
        model = CharBiLSTM(vocab_size=vocab.size, emb_dim=embed_dim)
    else:
        raise ValueError("model_type must be 'char_cnn' or 'bilstm'")

    device = torch.device("cpu")
    model.to(device)

    train_ds = URLDataset(train_urls, train_labels)
    test_ds = URLDataset(test_urls, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    start_train = time.time()
    model.train()
    for _ in range(int(epochs)):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    training_time_sec = float(time.time() - start_train)

    start_inf = time.time()
    model.eval()
    probs_list: List[float] = []
    y_list: List[int] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.extend([float(p) for p in probs])
            y_list.extend([int(v) for v in yb.numpy().tolist()])
    inference_time_sec = float(time.time() - start_inf)

    y_true = np.asarray(y_list, dtype=int)
    y_prob_pos1 = np.asarray(probs_list, dtype=float)
    y_pred = (y_prob_pos1 >= float(threshold)).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob_pos1)),
        "training_time_sec": training_time_sec,
        "inference_time_sec": inference_time_sec,
        "latency_per_url_ms": float(1000.0 * inference_time_sec / max(1, len(y_true))),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist(),
        },
    }

    return metrics


def main() -> int:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    defaults = _default_paths(base_dir)

    parser = argparse.ArgumentParser(description="R2-3: DL baseline (Char-CNN or BiLSTM) for raw URL classification. Outputs JSON only.")
    parser.add_argument("--white-list", default=defaults["white_list"])
    parser.add_argument("--block-list", default=defaults["block_list"])
    parser.add_argument("--output", default=defaults["output"])

    parser.add_argument("--model-type", choices=["char_cnn", "bilstm"], default="char_cnn")
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--max-vocab-size", type=int, default=0, help="0 means no limit")
    parser.add_argument("--embed-dim", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.3)

    args = parser.parse_args()

    df_w = pd.read_csv(args.white_list)
    df_b = pd.read_csv(args.block_list)
    for df in (df_w, df_b):
        if "url" not in df.columns or "label" not in df.columns:
            raise KeyError("white_list.csv/block_list.csv must contain 'url' and 'label' columns")

    df = pd.concat([df_w[["url", "label"]], df_b[["url", "label"]]], ignore_index=True)
    urls = df["url"].astype(str).map(lambda s: s.strip()).tolist()
    labels = df["label"].astype(int).to_numpy()

    train_urls, test_urls, y_train, y_test = train_test_split(
        urls,
        labels,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=labels,
    )

    max_vocab_size: Optional[int]
    if int(args.max_vocab_size) <= 0:
        max_vocab_size = None
    else:
        max_vocab_size = int(args.max_vocab_size)

    vocab = _build_vocab(train_urls, max_vocab_size=max_vocab_size)

    if not _torch_import():
        raise ImportError(
            "PyTorch is required for this script but is not installed. "
            "Please install torch (CPU version is sufficient) and rerun."
        )

    metrics = _train_eval_torch(
        model_type=str(args.model_type),
        train_urls=train_urls,
        train_labels=y_train,
        test_urls=test_urls,
        test_labels=y_test,
        vocab=vocab,
        max_len=int(args.max_len),
        embed_dim=int(args.embed_dim),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        seed=int(args.seed),
        threshold=float(args.threshold),
    )

    out: Dict = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model_type": str(args.model_type),
            "label_convention": {"0": "Malicious", "1": "Benign"},
            "threshold": float(args.threshold),
            "data": {
                "white_list": os.path.relpath(args.white_list, base_dir),
                "block_list": os.path.relpath(args.block_list, base_dir),
                "train_size": int(len(train_urls)),
                "test_size": int(len(test_urls)),
                "test_split_ratio": float(args.test_size),
                "seed": int(args.seed),
            },
            "preprocess": {
                "max_len": int(args.max_len),
                "vocab_size": int(vocab.size),
                "max_vocab_size": int(args.max_vocab_size),
                "embed_dim": int(args.embed_dim),
            },
            "train": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
            },
        },
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
