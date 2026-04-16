"""Duplicate group formation using Union-Find for all detection stages."""

from __future__ import annotations

from rich.console import Console

from dedup.db import Database
from dedup.perceptual import _from_signed_int64


class UnionFind:
    """Disjoint Set Union for forming connected components from pairwise matches."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def groups(self) -> dict[int, list[int]]:
        """Return a mapping of root -> list of members."""
        result: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            result.setdefault(root, []).append(i)
        return result


def find_exact_duplicates(db: Database) -> None:
    """Find exact duplicate groups from SHA256 hashes."""
    console = Console()

    db.clear_groups("exact")

    duplicates = db.get_exact_duplicates()
    if not duplicates:
        console.print("[dim]No exact duplicates found.[/dim]")
        return

    group_count = 0
    for row in duplicates:
        images = db.get_images_by_hash(row["sha256"])
        if len(images) < 2:
            continue

        group_id = db.insert_duplicate_group("exact", 1.0, len(images))
        group_count += 1

        # Create pairs for all combinations in the group
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                db.insert_duplicate_pair(
                    group_id, images[i]["id"], images[j]["id"], 1.0
                )

    db.conn.commit()
    console.print(f"[green]Found {group_count} exact duplicate groups[/green]")


def _hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two 64-bit integers."""
    return bin(a ^ b).count("1")


def find_perceptual_duplicates(db: Database, threshold: int = 10) -> None:
    """Find perceptual duplicate groups using Hamming distance on pHash values."""
    console = Console()

    db.clear_groups("perceptual")

    all_hashes = db.get_all_phashes()
    n = len(all_hashes)

    if n == 0:
        console.print("[dim]No perceptual hashes computed yet.[/dim]")
        return

    console.print(f"[bold]Comparing {n} perceptual hashes[/bold] (threshold={threshold})")

    # Build index mapping: position -> image_id
    # Convert signed int64 back to unsigned for Hamming distance computation
    ids = [row["id"] for row in all_hashes]
    phash_ints = [_from_signed_int64(row["phash_int"]) for row in all_hashes]

    uf = UnionFind(n)
    pair_similarities: dict[tuple[int, int], float] = {}

    # All-pairs comparison using Hamming distance
    # For 500K images this is O(n^2) which is expensive.
    # Optimization: bucket by first 8 bits of hash, compare within/between adjacent buckets.
    buckets: dict[int, list[int]] = {}
    for idx, ph in enumerate(phash_ints):
        if ph is None:
            continue
        key = ph >> 56  # top 8 bits
        buckets.setdefault(key, []).append(idx)

    # Compare within each bucket and adjacent buckets
    sorted_keys = sorted(buckets.keys())
    for ki, key in enumerate(sorted_keys):
        bucket = buckets[key]
        # Within-bucket comparisons
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                idx_a, idx_b = bucket[i], bucket[j]
                dist = _hamming_distance(phash_ints[idx_a], phash_ints[idx_b])
                if dist <= threshold:
                    uf.union(idx_a, idx_b)
                    similarity = 1.0 - (dist / 64.0)
                    pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
                    pair_similarities[pair_key] = similarity

        # Cross-bucket comparisons with adjacent keys
        for ki2 in range(ki + 1, len(sorted_keys)):
            other_key = sorted_keys[ki2]
            # If top-8 bits differ by more than threshold, hashes can't match
            if _hamming_distance(key, other_key) > threshold:
                # Keys are sorted, but Hamming distance isn't monotonic with integer value.
                # We still check all adjacent buckets to avoid misses.
                # However, for efficiency, break if too far apart in value.
                if other_key - key > 255:
                    break
                continue

            other_bucket = buckets[other_key]
            for idx_a in bucket:
                for idx_b in other_bucket:
                    dist = _hamming_distance(phash_ints[idx_a], phash_ints[idx_b])
                    if dist <= threshold:
                        uf.union(idx_a, idx_b)
                        similarity = 1.0 - (dist / 64.0)
                        pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
                        pair_similarities[pair_key] = similarity

    # Convert Union-Find groups to database groups
    group_count = 0
    for members in uf.groups().values():
        if len(members) < 2:
            continue

        # Compute average similarity for the group
        group_sims = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair_key = (min(members[i], members[j]), max(members[i], members[j]))
                sim = pair_similarities.get(pair_key, 0.0)
                group_sims.append(sim)

        avg_confidence = sum(group_sims) / len(group_sims) if group_sims else 0.0

        group_id = db.insert_duplicate_group("perceptual", avg_confidence, len(members))
        group_count += 1

        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair_key = (min(members[i], members[j]), max(members[i], members[j]))
                sim = pair_similarities.get(pair_key, 0.0)
                db.insert_duplicate_pair(group_id, ids[members[i]], ids[members[j]], sim)

    db.conn.commit()
    console.print(f"[green]Found {group_count} perceptual duplicate groups[/green]")


def find_cnn_duplicates(db: Database, threshold: float = 0.92) -> None:
    """Find CNN-based duplicate groups using FAISS similarity search."""
    import numpy as np

    console = Console()

    db.clear_groups("cnn")

    n = db.get_embeddings_count()

    if n == 0:
        console.print("[dim]No embeddings computed yet.[/dim]")
        return

    console.print(f"[bold]Comparing {n} embeddings[/bold] (threshold={threshold})")

    # Determine embedding dimension from the first row without holding all rows in memory
    first_row = next(iter(db.get_all_embeddings()))
    dim = len(np.frombuffer(first_row["embedding"], dtype=np.float32))

    # Stream embeddings into a pre-allocated array to avoid holding sqlite3.Row objects
    ids: list[int] = []
    embeddings = np.empty((n, dim), dtype=np.float32)
    for i, row in enumerate(db.get_all_embeddings()):
        ids.append(row["id"])
        embeddings[i] = np.frombuffer(row["embedding"], dtype=np.float32)

    # Use FAISS for efficient similarity search
    try:
        import faiss

        index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity on L2-normalized vectors
        index.add(embeddings)

        k = min(50, n)  # Search for top-50 neighbors
        # Chunk the search to limit peak result-matrix memory
        search_chunk = 10_000
        all_sims = []
        all_idxs = []
        for start in range(0, n, search_chunk):
            chunk = embeddings[start : start + search_chunk]
            s, ix = index.search(chunk, k)
            all_sims.append(s)
            all_idxs.append(ix)
        similarities = np.vstack(all_sims)
        indices = np.vstack(all_idxs)
    except ImportError:
        # Fallback to numpy if FAISS not available
        if n > 50_000:
            console.print(
                f"[red]FAISS is not installed and {n:,} embeddings is too large for the "
                f"numpy fallback (would require ~{n * n * 4 // 1_073_741_824:.0f} GB RAM). "
                "Install faiss-cpu: pip install faiss-cpu[/red]"
            )
            return
        console.print("[yellow]FAISS not installed, using numpy (slower).[/yellow]")
        similarities_matrix = embeddings @ embeddings.T
        k = min(50, n)
        indices = np.argsort(-similarities_matrix, axis=1)[:, :k]
        similarities = np.take_along_axis(similarities_matrix, indices, axis=1)

    # Build Union-Find from FAISS results
    uf = UnionFind(n)
    pair_similarities: dict[tuple[int, int], float] = {}

    for i in range(n):
        for j_pos in range(k):
            j = int(indices[i][j_pos])
            sim = float(similarities[i][j_pos])
            if i == j or sim < threshold:
                continue
            uf.union(i, j)
            pair_key = (min(i, j), max(i, j))
            if pair_key not in pair_similarities or sim > pair_similarities[pair_key]:
                pair_similarities[pair_key] = sim

    # Convert to database groups
    group_count = 0
    for members in uf.groups().values():
        if len(members) < 2:
            continue

        group_sims = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair_key = (min(members[i], members[j]), max(members[i], members[j]))
                sim = pair_similarities.get(pair_key, threshold)
                group_sims.append(sim)

        avg_confidence = sum(group_sims) / len(group_sims) if group_sims else threshold

        group_id = db.insert_duplicate_group("cnn", avg_confidence, len(members))
        group_count += 1

        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair_key = (min(members[i], members[j]), max(members[i], members[j]))
                sim = pair_similarities.get(pair_key, threshold)
                db.insert_duplicate_pair(group_id, ids[members[i]], ids[members[j]], sim)

    db.conn.commit()
    console.print(f"[green]Found {group_count} CNN duplicate groups[/green]")
