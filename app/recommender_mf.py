import pickle

import faiss
import numpy as np
import torch

from app.ml.matrix_factorization import MatrixFactorization


class MFRecommender:

    def __init__(
        self,
        model_path="models/ranker_mf_model.pt",
        index_path="models/ranker_item_index.faiss",
        mappings_path="models/ranker_mf_mappings.pkl",
    ):
        # -----------------------------------------------------
        # Load mappings
        # -----------------------------------------------------
        with open(mappings_path, "rb") as f:
            mappings = pickle.load(f)

        self.user_to_idx = mappings["user_to_idx"]
        self.item_to_idx = mappings["item_to_idx"]

        self.idx_to_item = {
            idx: item_id
            for item_id, idx in self.item_to_idx.items()
        }

        # -----------------------------------------------------
        # Load MF model
        # -----------------------------------------------------
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

        self.embedding_dim = checkpoint[
            "embedding_dim"
        ]

        self.model = MatrixFactorization(
            num_users=len(self.user_to_idx),
            num_items=len(self.item_to_idx),
            embedding_dim=self.embedding_dim,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # -----------------------------------------------------
        # Load FAISS index
        # -----------------------------------------------------
        self.index = faiss.read_index(
            index_path
        )

    def recommend(
        self,
        user_id,
        interacted_items,
        k=10,
        candidate_count=500,
    ):
        """
        Generate recommendations using:

            User embedding
                  ↓
              FAISS Top-N
                  ↓
             Filter history
                  ↓
                Top-K
        """

        if user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[user_id]

        # -----------------------------------------------------
        # Get user embedding
        # -----------------------------------------------------
        with torch.no_grad():

            user_tensor = torch.tensor(
                [user_idx],
                dtype=torch.long,
            )

            user_embedding = (
                self.model.user_embedding(
                    user_tensor
                )
                .squeeze(0)
                .numpy()
                .astype(np.float32)
            )

        # -----------------------------------------------------
        # Retrieve candidates
        #
        # Request extra candidates because some will be
        # filtered as previously interacted items.
        # -----------------------------------------------------
        search_k = min(
            candidate_count + len(interacted_items),
            self.index.ntotal,
        )

        query = user_embedding.reshape(
            1,
            -1,
        )

        _, indices = self.index.search(
            query,
            search_k,
        )

        # -----------------------------------------------------
        # Filter previously interacted items
        # -----------------------------------------------------
        recommendations = []

        for item_idx in indices[0]:

            item_idx = int(item_idx)

            if item_idx not in self.idx_to_item:
                continue

            item_id = self.idx_to_item[
                item_idx
            ]

            if item_id in interacted_items:
                continue

            recommendations.append(item_id)

            if len(recommendations) >= k:
                break

        return recommendations
