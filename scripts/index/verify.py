"""Verify shipped FAISS artefacts without downloading the large GGUF model.

This catches a stale or partial index push, corrupted vector mapping and
incorrect manifest versions; it does not assess semantic answer quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.catalog.corpus import corpus_digest
from app.catalog.index import load_artifacts
from app.pipeline.chunks import CHUNK_VERSION


def verify_index(root: Path, *, require_current_chunking: bool = False) -> dict[str, int]:
    index, _, chunks, manifest, config = load_artifacts(root, require_provenance=False)
    if require_current_chunking:
        if not config:
            raise ValueError("Missing chunk manifest")
        if config.get("chunk_version") != CHUNK_VERSION or manifest.get("chunk_config") != config:
            raise ValueError(f"Rebuild the index with chunking version {CHUNK_VERSION}")
        if manifest.get("corpus_sha256") != corpus_digest(root.parent):
            raise ValueError("Index does not match the canonical corpus")
    return {"chunks": len(chunks), "dimensions": index.d, "vectors": index.ntotal}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("--require-current-chunking", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_index(args.index_dir, require_current_chunking=args.require_current_chunking)))


if __name__ == "__main__":
    main()
