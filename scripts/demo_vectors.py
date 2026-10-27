"""Chunk 3 demo: embeddings + FAISS retrieval, fully offline.

    python scripts/demo_vectors.py          # mock embedder (default, free)
    python scripts/demo_vectors.py --real   # all-MiniLM-L6-v2 (downloads once)

What it shows
-------------
1. Texts -> vectors (embedding).
2. Top-k cosine neighbors for a query (vector search).
3. Duplicate-id handling (re-add is skipped).
4. Save/load round trip (ids survive restarts).

It deliberately shows NO relationship types: similarity only proposes
candidates; SUPPORTS/CONTRADICTS come from the Chunk-4 classifier.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from latent_workspace.config import get_settings
from latent_workspace.embeddings import EmbeddingError, build_embedder
from latent_workspace.vector import FaissVectorStore, build_vector_store

CORPUS = [
    "iu_0001  Rising electricity costs pushed factory output prices higher",
    "iu_0002  Power tariffs increased sharply for industrial consumers",
    "iu_0003  The championship final was decided in a penalty shootout",
    "iu_0004  Goalkeepers saved two penalties in extra time",
    "iu_0005  New telescope images revealed a distant galaxy cluster",
    "iu_0006  Astronomers observed light emitted billions of years ago",
    "iu_0007  Shipping container rates fell back to pre-pandemic levels",
    "iu_0008  Freight volumes normalised after the supply-chain backlog",
]
QUERY = "Factories faced higher electricity and power costs"


def line(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="DRLW vector retrieval demo")
    parser.add_argument(
        "--real",
        action="store_true",
        help="use the real all-MiniLM-L6-v2 model instead of the mock embedder",
    )
    args = parser.parse_args()

    settings = get_settings()
    try:
        if args.real:
            from latent_workspace.embeddings import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(
                model_name=settings.embedding_model,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
            )
        else:
            embedder = build_embedder(settings)  # mock when DLRW_MOCK_MODE=true
    except EmbeddingError as exc:
        print(f"Could not create embedder: {exc}")
        return 1

    line("SETUP")
    print(f"embedder : {embedder.model_name} (dim={embedder.dim})")
    print(f"corpus   : {len(CORPUS)} texts")

    store = build_vector_store(embedder, settings)
    texts = [t.split("  ", 1)[1] for t in CORPUS]
    ids = [t.split("  ", 1)[0] for t in CORPUS]

    line("1) EMBED + INDEX")
    vectors = embedder.embed(texts)
    added = store.add(ids, vectors)
    print(f"embedded {len(vectors)} texts -> {len(vectors[0])}-dim vectors")
    print(f"indexed {added} vectors (of {len(ids)} requested)")

    line("2) TOP-K SEARCH  (query below)")
    print(f'   query: "{QUERY}"')
    hits = store.search(embedder.embed([QUERY])[0], top_k=4)
    for rank, hit in enumerate(hits, 1):
        label = texts[ids.index(hit.id)][:60]
        print(f"   {rank}. {hit.id}  cos={hit.score:+.4f}  {label}")

    line("3) DUPLICATE HANDLING")
    again = store.add([ids[0]], embedder.embed([texts[0]]))
    print(f"re-adding '{ids[0]}' -> added={again} (0 = skipped, first wins), size={store.size}")

    line("4) SAVE / LOAD PERSISTENCE")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vectors.faiss"
        store.save(path)
        restored = FaissVectorStore.load(path)
        print(f"saved to {path.name} (+ {path.name}.ids.json sidecar)")
        print(f"restored size={restored.size}, dim={restored.dim}")
        reloaded_hits = restored.search(embedder.embed([QUERY])[0], top_k=4)
        same = [h.as_tuple() for h in hits] == [h.as_tuple() for h in reloaded_hits]
        print(f"identical results after reload: {same}")

    line("NOTE")
    print("Scores above are COSINE SIMILARITY only - 'worth comparing'.")
    print("No SUPPORTS/CONTRADICTS decisions were made from these scores.")
    print("OK - chunk 3 demo finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
