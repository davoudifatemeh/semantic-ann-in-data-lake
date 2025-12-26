import hnswlib
import numpy as np
import pickle

class ANNIndex:
    def __init__(self, dim: int, space: str = "l2"):
        self.index = hnswlib.Index(space=space, dim=dim)
        self.space = space
        self.ids = []  # ["table::col"]

    def build(self, embeddings: np.ndarray, ids: list[str],
              ef_construction: int = 200, M: int = 16, ef: int = 50):
        assert embeddings.shape[0] == len(ids), "Embeddings and ids must match"
        self.index.init_index(max_elements=embeddings.shape[0],
                              ef_construction=ef_construction, M=M)
        self.index.add_items(embeddings, list(range(len(ids))))
        self.ids = ids
        self.index.set_ef(ef)
        print(f"ANNIndex: Built index with {len(self.ids)} items, space={self.space}, dim={embeddings.shape[1]}")

    def query(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        # Query one embedding and return (id, distance) pairs.
        labels, distances = self.index.knn_query(query_embedding, k=k)
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            results.append((self.ids[idx], float(dist)))
        # print(f"ANNIndex: Query returned {len(results)} results")
        return results

    def batch_query(self, query_embeddings: np.ndarray, k: int = 5) -> list[list[tuple[str, float]]]:
        # Query multiple embeddings at once.
        # Returns: list of result lists, one per query embedding.
        labels, distances = self.index.knn_query(query_embeddings, k=k)
        all_results = []
        for row_labels, row_distances in zip(labels, distances):
            results = []
            for idx, dist in zip(row_labels, row_distances):
                results.append((self.ids[idx], float(dist)))
            all_results.append(results)
        return all_results

    def save(self, path: str):
        self.index.save_index(path + ".bin")
        with open(path + ".ids", "wb") as f:
            pickle.dump(self.ids, f)

    def load(self, path: str):
        self.index.load_index(path + ".bin")
        with open(path + ".ids", "rb") as f:
            self.ids = pickle.load(f)
        print(f"ANNIndex: Loaded index with {len(self.ids)} items from {path}")