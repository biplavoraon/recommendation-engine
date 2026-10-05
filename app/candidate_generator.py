import faiss
import numpy as np


class FAIISCandidateGenerator:
    def __init__(self, model, index_path, item_to_idx):
        self.model = model
        self.model.eval()

        self.index = faiss.read_index(index_path)

        self.item_to_idx = item_to_idx
        self.idx_to_item = {
            idx: item_id
            for item_id, idx in item_to_idx.items()
        }

    def generate(
        self,
        user_idx,
        seen_items,
        num_candidates=100,
    ):
        # Retrieve a larger pool than the final candidate count.
        search_k = min(
            num_candidates * 2 + len(seen_items),
            self.index.ntotal,
        )

        user_embedding = (
            self.model.user_embedding.weight[user_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        query = user_embedding.reshape(1, -1)

        _, indices = self.index.search(
            query,
            search_k,
        )

        candidate_indices = []

        for item_idx in indices[0]:
            if item_idx == -1:
                continue

            item_id = self.idx_to_item[item_idx]

            if item_id in seen_items:
                continue

            candidate_indices.append(item_idx)

        if not candidate_indices:
            return []

        # Convert candidate indices to a tensor.
        import torch

        item_tensor = torch.tensor(
            candidate_indices,
            dtype=torch.long,
        )

        user_tensor = torch.tensor(
            [user_idx],
            dtype=torch.long,
        )

        with torch.no_grad():
            user_embedding = self.model.user_embedding(
                user_tensor
            )

            item_embeddings = self.model.item_embedding(
                item_tensor
            )

            embedding_scores = (
                item_embeddings @ user_embedding.squeeze(0)
            )

            item_biases = self.model.item_bias(
                item_tensor
            ).squeeze(-1)

            scores = embedding_scores + item_biases

        # Rerank using the full item-dependent MF score.
        order = torch.argsort(
            scores,
            descending=True,
        )

        selected = [
            self.idx_to_item[candidate_indices[i]]
            for i in order[:num_candidates]
        ]

        return selected
