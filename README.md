# Scalable Recommendation Engine

An end-to-end recommendation system built with **PyTorch, FAISS, FastAPI, PostgreSQL, Redis, and Docker**.

The system learns personalized item embeddings using **Matrix Factorization with Bayesian Personalized Ranking (BPR)**, retrieves candidates using **FAISS**, serves recommendations through a **FastAPI** service, and uses **Redis caching** to reduce online serving latency.

## Architecture

```text
                         OFFLINE
                            │
                            ▼
                  ┌──────────────────┐
                  │  MovieLens 100K  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Data Preparation │
                  │ Temporal Splits  │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Matrix Factor.   │
                  │      + BPR       │
                  └────────┬─────────┘
                           │
                     Item embeddings
                           │
                           ▼
                  ┌──────────────────┐
                  │      FAISS       │
                  │ Candidate Search │
                  └────────┬─────────┘
                           │
                           │
                        ONLINE
                           │
                           ▼
                  ┌──────────────────┐
                  │     FastAPI      │
                  └────────┬─────────┘
                           │
                     Cache lookup
                           │
                    ┌──────┴──────┐
                    │             │
                   HIT           MISS
                    │             │
                    ▼             ▼
                 ┌───────┐   ┌────────────┐
                 │ Redis │   │ MF + FAISS │
                 └───┬───┘   └──────┬─────┘
                     │              │
                     └──────┬───────┘
                            ▼
                       Top-K Items
```

## Features

- Personalized recommendation using Matrix Factorization
- BPR loss for implicit-feedback learning
- Temporal train/validation/test splitting
- FAISS-based approximate/exact candidate retrieval
- PostgreSQL data layer
- Redis recommendation caching
- FastAPI inference API
- Dockerized deployment
- Offline ranking evaluation with Recall@K and NDCG@K
- API latency and throughput benchmarking

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| ML | PyTorch |
| Recommendation model | Matrix Factorization + BPR |
| Candidate retrieval | FAISS |
| Database | PostgreSQL |
| Cache | Redis |
| API | FastAPI |
| Containerization | Docker / Docker Compose |
| Dataset | MovieLens 100K |

---

## Recommendation Pipeline

The serving pipeline consists of two main stages.

### 1. Candidate Generation

Each user is represented by a learned latent embedding:

\[
p_u \in \mathbb{R}^{64}
\]

and each item by:

\[
q_i \in \mathbb{R}^{64}
\]

FAISS performs maximum inner-product search over item embeddings to retrieve a candidate set.

For a user \(u\) and item \(i\), the underlying MF model scores:

\[
\hat{r}_{ui}
=
p_u^\top q_i
+
b_u
+
b_i
+
b_0
\]

The system retrieves up to 500 candidates before filtering previously interacted items.

### 2. Top-K Recommendation

The candidate items are filtered against the user's interaction history and the highest-ranked unseen items are returned.

---

## Model Training

The system uses **Bayesian Personalized Ranking (BPR)** because MovieLens interactions can be treated as implicit preference signals.

For each observed interaction:

```text
(user, positive item, negative item)
```

the training objective encourages:

\[
\hat{r}_{u,i^+} > \hat{r}_{u,i^-}
\]

using the pairwise loss:

\[
\mathcal{L}_{BPR}
=
-\log
\sigma
\left(
\hat{r}_{u,i^+}
-
\hat{r}_{u,i^-}
\right)
\]

Training configuration:

```text
Embedding dimension: 64
Optimizer:           Adam
Learning rate:       0.001
Batch size:          1024
Epochs:              20
Weight decay:        1e-6
```

---

## Dataset

The project uses the **MovieLens 100K** dataset:

- 943 users
- 1,682 items
- 100,000 interactions

The evaluation uses temporal splitting so that future interactions are not used to predict the past.

For the final ranking evaluation, the development data is further divided temporally into training, validation, and test portions.

This prevents information from future interactions from leaking into model training.

---

## Model Evaluation

The initial popularity baseline achieved:

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| Popularity | 0.0753 | 0.0394 |
| Matrix Factorization | 0.0870 | 0.0420 |

The temporally trained MF model used for the serving pipeline achieved:

| Metric | Score |
|---|---:|
| Recall@10 | **0.1004** |
| NDCG@10 | **0.0486** |
| Candidate Recall@500 | **0.8876** |

Candidate Recall@500 of 88.76% means that approximately 88.8% of validation target items were present in the retrieved candidate set.

Therefore, candidate generation places an upper bound on the recall achievable by any downstream ranking model.

---

## Ranking Experiments

A neural ranking stage was also investigated using user statistics, item statistics, MF scores, and learned embeddings.

Several ranking configurations were evaluated, including:

- Hard-negative sampling
- Random-negative sampling
- Listwise ranking
- Rank-aware negative sampling
- Linear ranking

The neural rankers achieved strong training-set performance but failed to generalize to the full 500-candidate validation distribution.

For example, the first neural ranker achieved:

```text
Training Recall@10:     0.8799
Training NDCG@10:       0.7626

Validation Recall@10:   0.0048
Validation NDCG@10:     0.0019
```

The experiments demonstrated substantial overfitting and candidate-distribution mismatch.

The final system therefore uses **MF + FAISS** rather than forcing a downstream neural ranker that performs worse on held-out data.

This is an intentional model-selection decision based on validation performance.

---

## API

### Health Check

```http
GET /health
```

Example:

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok"
}
```

### Get Recommendations

```http
GET /recommend/{user_id}?k=10
```

Example:

```bash
curl "http://127.0.0.1:8000/recommend/1?k=10"
```

Example response:

```json
{
  "user_id": 1,
  "recommendations": [
    423,
    318,
    496,
    403,
    405,
    568,
    739,
    357,
    732,
    474
  ]
}
```

The API accepts:

```text
1 <= k <= 100
```

Previously interacted items are filtered from the recommendations.

---

## Redis Caching

Recommendation responses are cached using:

```text
recommendations:{user_id}:{k}
```

with a TTL of 300 seconds.

The serving flow is:

```text
Request
   │
   ▼
Redis lookup
   │
   ├── HIT ──────► Return cached recommendations
   │
   └── MISS
         │
         ▼
      MF + FAISS
         │
         ▼
      Cache result
         │
         ▼
      Return result
```

---

## Performance Benchmark

The API was benchmarked using 100 sequential requests from the local host.

### Serving Performance

| Configuration | Mean | P50 | P95 | P99 | Throughput |
|---|---:|---:|---:|---:|---:|
| API without Redis | 2.255 ms | 1.961 ms | 2.427 ms | 27.317 ms | 443.5 req/s |
| API + Redis | 0.893 ms | 0.814 ms | 1.339 ms | 3.609 ms | 1119.9 req/s |
| Docker + Redis | **1.047 ms** | **0.997 ms** | **1.942 ms** | **3.309 ms** | **955.5 req/s** |

Compared with the original non-cached API, the final Docker deployment achieved:

- **53.6% lower mean latency**
- **49.2% lower P50 latency**
- **87.9% lower P99 latency**
- **2.15× higher throughput**

These measurements are from a local development environment and should not be interpreted as production capacity.

---

## Project Structure

```text
recommendation-engine/
│
├── app/
│   ├── config.py
│   ├── db.py
│   ├── models.py
│   ├── main.py
│   ├── recommender.py
│   ├── recommender_mf.py
│   ├── evaluation.py
│   │
│   └── ml/
│       ├── dataset.py
│       ├── losses.py
│       ├── matrix_factorization.py
│       ├── features.py
│       ├── ranker.py
│       ├── ranker_losses.py
│       └── ranker_dataset.py
│
├── data/
│   └── ml-100k/
│
├── models/
│   ├── ranker_mf_model.pt
│   ├── ranker_item_index.faiss
│   └── ...
│
├── scripts/
│   ├── load_movielens.py
│   ├── train_mf.py
│   ├── evaluate_mf.py
│   ├── build_faiss_index.py
│   ├── evaluate_candidates.py
│   ├── build_features.py
│   ├── train_ranker.py
│   ├── evaluate_ranker_validation.py
│   └── benchmark_api.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd recommendation-engine
```

### 2. Start PostgreSQL and Redis

```bash
docker compose up -d postgres redis
```

### 3. Install Python dependencies

```bash
python -m venv reco_venv
source reco_venv/bin/activate

pip install -r requirements.txt
```

### 4. Create database tables

```bash
python scripts/create_tables.py
```

### 5. Load MovieLens

```bash
python scripts/load_movielens.py
```

### 6. Run the API

```bash
uvicorn app.main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

---

## Running with Docker

The complete application can be started using Docker Compose:

```bash
docker compose up -d --build
```

Check the services:

```bash
docker compose ps
```

Expected services:

```text
recommendation-api
recommendation-postgres
recommendation-redis
```

Test the API:

```bash
curl http://127.0.0.1:8000/health
```

Then:

```bash
curl "http://127.0.0.1:8000/recommend/1?k=10"
```

View API logs:

```bash
docker compose logs api
```

Stop the stack:

```bash
docker compose down
```

The PostgreSQL data is persisted using a Docker named volume.

---

## Benchmarking

The API benchmark can be run with:

```bash
python scripts/benchmark_api.py
```

It reports:

```text
Requests
Mean latency
P50 latency
P95 latency
P99 latency
Throughput
```

---

## Design Decisions

### Why Matrix Factorization?

Matrix Factorization provides a simple and strong baseline for collaborative filtering while producing compact user and item embeddings suitable for approximate nearest-neighbor retrieval.

### Why BPR?

The dataset represents user-item interactions rather than explicit positive/negative preference labels. BPR is therefore appropriate for optimizing relative preference between observed and unobserved items.

### Why FAISS?

Scoring every item for every request becomes expensive as the catalog grows. FAISS provides an efficient vector-search interface for retrieving candidate items from learned embeddings.

### Why Redis?

Recommendations for a user can be reused for repeated requests. Redis reduces repeated computation and database access for frequently requested recommendation lists.

### Why Docker?

Docker provides reproducible deployment of the API, database, and cache as a single application stack.

### Why not use the neural ranker?

The downstream ranking experiments showed strong training performance but poor held-out generalization on the full candidate set. The simpler MF retrieval pipeline performed more reliably on the validation data and was therefore selected for the final system.

---

## Future Improvements

Potential extensions include:

- Larger-scale interaction datasets
- Streaming interaction ingestion
- Incremental model updates
- User/item cold-start handling
- Two-tower retrieval models
- Feature normalization and richer contextual features
- A better-calibrated learning-to-rank stage
- ANN benchmarking on a substantially larger catalog
- Concurrent load testing with Locust or k6
- Recommendation cache invalidation when new interactions arrive
- Monitoring and observability for production deployment

## License

This project is intended as a research and engineering portfolio project.
