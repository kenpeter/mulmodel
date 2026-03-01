from __future__ import annotations
import io
import json
import os
import urllib.request
import zipfile

# Local cache path
_CACHE_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "corpus_cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "sentiment.json")

# UCI Sentiment Labelled Sentences dataset
# 2748 real reviews from Amazon / IMDB / Yelp — tab-separated, label 0/1
_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00331/sentiment%20labelled%20sentences.zip"
)
_UCI_FILES = [
    "sentiment labelled sentences/amazon_cells_labelled.txt",
    "sentiment labelled sentences/imdb_labelled.txt",
    "sentiment labelled sentences/yelp_labelled.txt",
]

# ── Built-in fallback corpus (used when offline) ──────────────────────────────
_BUILTIN: list[tuple[str, float]] = [
    # Positive
    ("great product loved every minute of it", 1.0),
    ("excellent quality exceeded my expectations", 1.0),
    ("amazing experience would highly recommend", 1.0),
    ("fantastic service very happy with purchase", 1.0),
    ("wonderful movie beautiful story outstanding acting", 1.0),
    ("best purchase I have ever made works perfectly", 1.0),
    ("superb quality incredibly well made", 1.0),
    ("brilliant film loved the characters", 1.0),
    ("very impressed with the performance", 1.0),
    ("outstanding results highly recommend to everyone", 1.0),
    ("perfect exactly what I was looking for", 1.0),
    ("love this product works great every time", 1.0),
    ("incredible value for money very satisfied", 1.0),
    ("top quality fast delivery very happy", 1.0),
    ("awesome experience will definitely buy again", 1.0),
    # Negative
    ("terrible product broke after one day", -1.0),
    ("awful experience never buying again", -1.0),
    ("horrible quality complete waste of money", -1.0),
    ("very disappointed not as described at all", -1.0),
    ("boring movie slow plot bad acting", -1.0),
    ("worst purchase ever do not buy this", -1.0),
    ("poor quality fell apart immediately", -1.0),
    ("dreadful service very rude staff", -1.0),
    ("totally useless does not work at all", -1.0),
    ("bad experience would not recommend to anyone", -1.0),
    ("disappointing results far below expectations", -1.0),
    ("cheap and nasty broke on first use", -1.0),
    ("appalling quality money completely wasted", -1.0),
    ("slow and unreliable terrible performance", -1.0),
    ("horrible smell and ugly design total rubbish", -1.0),
]


class CorpusLoader:
    """
    Loads real sentiment text data for training TextDataGenerator.

    Priority order:
      1. Local cache  (instant — already downloaded)
      2. HuggingFace datasets library  (if installed)
      3. UCI ML Repository download  (urllib, no extra deps)
      4. Built-in fallback  (always works, offline)

    The corpus is (text: str, label: float) pairs where
    label = +1.0 for positive, -1.0 for negative.
    """

    def load(self, max_examples: int = 3000) -> list[tuple[str, float]]:
        # 1. Local cache
        data = self._load_cache()
        if data:
            print(f"  [Corpus] Loaded {len(data)} examples from cache.")
            return data[:max_examples]

        # 2. HuggingFace datasets
        data = self._try_huggingface()
        if data:
            self._save_cache(data)
            print(f"  [Corpus] Downloaded {len(data)} examples via HuggingFace.")
            return data[:max_examples]

        # 3. UCI download
        data = self._try_uci()
        if data:
            self._save_cache(data)
            print(f"  [Corpus] Downloaded {len(data)} examples from UCI.")
            return data[:max_examples]

        # 4. Built-in fallback
        print(f"  [Corpus] Using built-in {len(_BUILTIN)} examples (offline fallback).")
        return _BUILTIN

    # ── backends ──────────────────────────────────────────────────────────────

    def _load_cache(self) -> list[tuple[str, float]] | None:
        if not os.path.exists(_CACHE_FILE):
            return None
        try:
            with open(_CACHE_FILE) as f:
                raw = json.load(f)
            return [(item[0], float(item[1])) for item in raw]
        except Exception:
            return None

    def _save_cache(self, data: list[tuple[str, float]]) -> None:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump(data, f)

    def _try_huggingface(self) -> list[tuple[str, float]] | None:
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("rotten_tomatoes", split="train")
            out = []
            for ex in ds:
                label = 1.0 if ex["label"] == 1 else -1.0
                out.append((ex["text"], label))
            return out
        except Exception:
            return None

    def _try_uci(self) -> list[tuple[str, float]] | None:
        try:
            print("  [Corpus] Downloading UCI Sentiment dataset...")
            req = urllib.request.Request(
                _UCI_URL,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_bytes = resp.read()

            zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
            out: list[tuple[str, float]] = []

            for fname in _UCI_FILES:
                try:
                    content = zf.read(fname).decode("utf-8", errors="ignore")
                except KeyError:
                    # Try without subdirectory prefix
                    fname_alt = fname.split("/")[-1]
                    content   = zf.read(fname_alt).decode("utf-8", errors="ignore")

                for line in content.splitlines():
                    line = line.strip()
                    if not line or "\t" not in line:
                        continue
                    text, label_str = line.rsplit("\t", 1)
                    label_str = label_str.strip()
                    if label_str not in ("0", "1"):
                        continue
                    label = 1.0 if label_str == "1" else -1.0
                    out.append((text.strip(), label))

            return out if out else None

        except Exception as e:
            print(f"  [Corpus] UCI download failed: {e}")
            return None


# Module-level singleton + cached corpus
_loader  = CorpusLoader()
_corpus: list[tuple[str, float]] | None = None


def get_corpus(max_examples: int = 3000) -> list[tuple[str, float]]:
    """Return the sentiment corpus, loading/downloading once then caching in memory."""
    global _corpus
    if _corpus is None:
        _corpus = _loader.load(max_examples)
    return _corpus
