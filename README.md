# ML System - Product Recommendation | Amazon Products

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C?style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-1.x-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Part%201%20Complete-2EA44F?style=for-the-badge)
![Deploy](https://img.shields.io/badge/Deploy-Part%202%20Planned-F5A623?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-Planned-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Planned-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-Planned-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Planned-2496ED?style=for-the-badge&logo=docker&logoColor=white)

<p align="center">
  <img src="https://github.com/RafaelGallo/ML_System_recommendation_products_amazon_products/blob/main/img/1d14e970-2bf1-4dc5-b128-e27b50b83517.png?raw=true" width="100%"/>
</p>

## Overview

This project builds a complete **Product Recommendation System** using the Amazon Product
Reviews dataset. Two complementary Machine Learning approaches were developed and evaluated:
**KNN with Cosine Similarity** (Content-Based Filtering) and **Neural Collaborative Filtering (NCF)**
with TensorFlow/Keras. This is Part 1 of a two-part project. Part 2 will cover full production
deployment using Docker, MLflow, FastAPI, and Airflow.

## Business Problem

Amazon hosts millions of products across hundreds of categories. With such a vast catalog,
customers often struggle to discover relevant products, leading to poor shopping experiences
and reduced conversion rates. This project addresses the question:

> Given a product a customer is currently viewing, which other products are most similar and relevant?

The system recommends products based on review content, average rating, and product metadata,
without requiring user login or purchase history.

## Dataset

| Field | Value |
|-------|-------|
| Source | Amazon Product Reviews Dataset — Kaggle |
| Reviews | ~1,600 |
| Unique products | ~54 |
| Unique users | 836 |
| Rating scale | 1 to 5 stars |
| Relevant products | 31 (avg_rating >= 4.0) |

## Project Structure

```
ML_System_recommendation_products_amazon_products/
├── img/                        
├── output/                     
├── notebooks/
│   ├── knn_recommendation.ipynb
│   └── ncf_recommendation.ipynb
├── models/
│   ├── knn_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── scaler.pkl
│   ├── feature_matrix.pkl
│   ├── ncf_model.keras
│   ├── user_encoder.pkl
│   └── product_encoder.pkl
├── README.md
└── requirements.txt
```

## Part 1 — Machine Learning Models

### Model 1 — KNN with Cosine Similarity

Content-Based Filtering using TF-IDF vectorization of review texts combined with normalized
numerical features (avg rating, helpful votes, review count). Optimal K selected via grid
search (K=2 to K=20) using F1@K as the selection criterion.

#### Pipeline

```
Review Text  →  TF-IDF (500 features, bigrams)  →  Sparse Matrix
Numerical    →  MinMaxScaler                     →  Sparse Matrix
                                                         ↓
                                              hstack (combined matrix)
                                                         ↓
                                         NearestNeighbors (metric=cosine)
                                                         ↓
                                              Top-K Similar Products
```

---

#### Optimal K Search

![KNN Optimal K Search](https://github.com/RafaelGallo/ML_System_recommendation_products_amazon_products/blob/main/img/1.png?raw=true)

The left chart shows all four metrics (Precision, Recall, F1, Hit Rate) across K values from 2 to 20.
Hit Rate reaches 1.0 at K=5 and stays constant, meaning every product always receives at least one
relevant recommendation. Precision stabilizes around 0.78-0.80 across all K values, showing
consistent quality. Recall and F1 grow steadily as K increases, with the best F1@K of 0.6304
achieved at K=20. The right chart zooms into the F1@K curve, confirming K=20 as the optimal
selection by F1 maximization. The curve grows linearly without saturation, indicating the model
benefits from larger K within this dataset size.

#### KNN Results

| Metric | Value |
|--------|-------|
| Best K | 20 |
| Precision@5 | 0.7630 |
| Recall@5 | 0.1231 |
| F1@5 | 0.2119 |
| Hit Rate@5 | 1.0000 |
| Mean Cosine Similarity | 0.5716 |

#### Metrics Table

![KNN Metrics Table](https://github.com/RafaelGallo/ML_System_recommendation_products_amazon_products/blob/main/img/6.png?raw=true)

Precision@5 of 0.7630 indicates that 76% of recommended products are truly relevant,
showing strong recommendation quality. Recall@5 of 0.1231 reveals a coverage limitation:
the model finds only 12% of all relevant products in the catalog, which is expected given
the small dataset size and sparse TF-IDF representation. Hit Rate of 1.0 confirms that
every product always has at least one relevant recommendation in its top-5 list.

#### Sensitivity Analysis

![KNN Metrics Sensitivity](https://github.com/RafaelGallo/ML_System_recommendation_products_amazon_products/blob/main/img/7.png?raw=true)

The sensitivity analysis across K values (1, 3, 5, 10) shows that Precision remains stable
between 0.68 and 0.79 regardless of K, confirming consistent recommendation quality. Recall
and F1 grow as K increases, suggesting the model benefits from larger neighbor counts for
catalog coverage. Hit Rate reaches 1.0 at K=10, meaning all products receive at least one
relevant recommendation when K is large enough. The trade-off between Precision and Recall
stabilizes at K=5, making it a practical operating point for production.

#### Similarity Distribution

![KNN Similarity Distribution](https://github.com/RafaelGallo/ML_System_recommendation_products_amazon_products/blob/main/img/3.png?raw=true)

The cosine similarity distribution across all products shows a concentration between 0.55
and 0.75, with a median of 0.6076 and an overall mean of 0.5716. This indicates that the
majority of products have moderately strong similarity with their nearest neighbors, validating
that the TF-IDF feature matrix captures meaningful product relationships. Products with
similarity below 0.4 represent catalog outliers with very distinct review language.

#### Product Search Example

![KNN Product Search](https://github.com/RafaelGallo/ML_System_recommendation_products_amazon_products/blob/main/img/2.png?raw=true)

The product search for "Amazon Fire TV" returns 5 matching products in the catalog, displaying
their ASIN, brand, categories, average rating, and review count. The search is case-insensitive
and matches partial product names, allowing flexible user queries. Products with low avg_rating
(below 1.5) are visible in the match list, demonstrating that the search returns all matches
before the recommendation engine filters by relevance.

#### Recommendation Output

![KNN Recommendations Chart](https://github.com/RafaelGallo/ML_System_recommendation_products_amazon_products/blob/main/img/4.png?raw=true)

For the query product "Kindle Fire HD 7", the left chart shows the top-5 recommended products
ranked by cosine similarity score. The highest similarity of 0.566 is achieved by "Kindle Fire
HDX 7", confirming that the model correctly identifies similar Amazon tablet products. The right
chart shows avg ratings of recommended products, with "Fire HD 7 Tablet" being the only product
above the relevance threshold of 4.0 (green bar). The remaining products fall below the threshold
(orange bars), reflecting the dataset's overall low average ratings caused by normalized rating aggregation.

#### Top 5 Similarity Scores

![KNN Similarity Scores](https://github.com/RafaelGallo/ML_System_recommendation_products_amazon_products/blob/main/img/5.png?raw=true)

The top-5 similarity scores for the "Kindle" query range from 0.440 to 0.566, all within a
narrow band. This indicates that the recommended products share similar vocabulary in their
reviews and belong to the same product family (Amazon tablets and e-readers). The gradient
color from light to dark blue reflects the ranking from most to least similar. The compact
similarity range suggests the TF-IDF space is well-structured for this product category.

### Model 2 — Neural Collaborative Filtering (NCF)

NCF learns dense 32-dimensional embeddings for users and products via an MLP architecture
(Dense 128 → 64 → 32 with Dropout). Product recommendations are generated by cosine similarity
between trained product embeddings extracted from the model after training.

#### Architecture

![NCF Architecture](https://github.com/RafaelGallo/ML_System_recommendation_products_amazon_products/blob/main/output/model.png?raw=true)

The NCF architecture takes two separate inputs: user index and product index. Each input passes
through a dedicated Embedding layer (output shape: None, 1, 32), learning a 32-dimensional dense
representation. Both embeddings are flattened and concatenated into a 64-dimensional vector,
which feeds into three Dense layers (128 → 64 → 32) with ReLU activations and Dropout layers
(0.3 and 0.2) for regularization. The final output layer uses sigmoid activation to predict a
normalized rating in the range [0, 1], which is then denormalized back to the original 1-5 scale.
The dual-branch structure allows the model to independently learn user preferences and product
characteristics before combining them for interaction prediction.

```
user_input  →  Embedding(32)  →  Flatten  ─┐
                                             Concatenate(64)
product_input → Embedding(32) →  Flatten  ─┘
                                             ↓
                                        Dense(128, relu)
                                        Dropout(0.3)
                                        Dense(64, relu)
                                        Dropout(0.2)
                                        Dense(32, relu)
                                             ↓
                                        Dense(1, sigmoid)
                                             ↓
                                     Predicted Rating [0,1]
```

---

#### NCF Training History

![NCF Training](https://github.com/RafaelGallo/ML_System_recommendation_products_amazon_products/blob/main/output/0.png?raw=true)

The training history shows the model converging within the defined epochs, with EarlyStopping
restoring the best weights and ReduceLROnPlateau adjusting the learning rate when validation
loss plateaued. The gap between training and validation loss is expected given the small dataset
size of 1.6k reviews, as the model has limited interaction data to generalize from.

#### NCF Results

| Metric | Value | Category |
|--------|-------|----------|
| RMSE | 0.9165 | Regression |
| MAE | 0.5380 | Regression |
| MSE | 0.8400 | Regression |
| R² | 0.1156 | Regression |
| MAPE | 19.54% | Regression |
| Precision | 0.9133 | Classification |
| Recall | 0.8950 | Classification |
| F1 Score | 0.9040 | Classification |

The regression metrics (RMSE 0.9165, MAE 0.5380) indicate an average prediction error of
approximately 0.9 stars on the 1-5 scale, which is acceptable for a dataset of this size.
The R² of 0.1156 reveals limited variance explanation in exact rating prediction, a known
limitation of collaborative filtering on sparse small datasets. The classification metrics
tell a different story: Precision of 0.9133 and Recall of 0.8950 yield an F1 Score of 0.9040,
demonstrating that the NCF is highly effective at identifying which products are truly relevant,
even when it cannot predict the exact rating with high accuracy.

## Model Comparison

| Criteria | KNN | NCF |
|----------|-----|-----|
| Precision | 0.7630 | 0.9133 |
| Recall | 0.1231 | 0.8950 |
| F1 Score | 0.2119 | 0.9040 |
| Hit Rate | 1.0000 | — |
| Infrastructure | Simple, no GPU | GPU recommended |
| Cold Start | Supported | Not supported |
| Interpretability | High | Low |
| Production role | Fallback | Main model |

NCF outperforms KNN across all classification metrics, especially in Recall (0.89 vs 0.12),
demonstrating significantly greater capacity to cover the relevant product catalog. KNN, despite
lower coverage metrics, offers relevant operational advantages: no GPU required, fully interpretable,
and handles cold start effectively for products with no interaction history.

The recommended production architecture combines both models: NCF as the main model and KNN
as fallback for products with no interaction history.

## Part 2 — Production Deploy (Planned)

Part 2 will implement a complete MLOps pipeline for production deployment.

### Planned Architecture

```
                        Part 2 — MLOps Pipeline
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Airflow                                                   │
│   Orchestration  →  Model Retrain  →  MLflow Tracking       │
│                                           ↓                 │
│                                    Model Registry           │
│                                           ↓                 │
│                                     FastAPI                 │
│                                    REST API                 │
│                                  /recommend                 │
│                                           ↓                 │
│                               Docker Container              │
│                              (prod deployment)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Part 2 Components

| Tool | Role |
|------|------|
| FastAPI | REST API to serve recommendations |
| MLflow | Experiment tracking and model registry |
| Airflow | Pipeline orchestration and scheduled retraining |
| Docker | Containerization for reproducible deployment |

### Planned Endpoints

```
POST /recommend/knn
     body: { "product_name": "Kindle Fire HD 7" }
     returns: top-K similar products

POST /recommend/ncf
     body: { "product_name": "Kindle Fire HD 7" }
     returns: top-K similar products by NCF embeddings

GET /health
     returns: API status

GET /metrics
     returns: current model evaluation metrics
```

## How to Run

```bash
# Clone the repository
git clone https://github.com/RafaelGallo/ML_System_recommendation_products_amazon_products.git
cd ML_System_recommendation_products_amazon_products

# Install dependencies
pip install -r requirements.txt

# Run on Kaggle
# Dataset path: /kaggle/input/datasets/yasserh/amazon-product-reviews-dataset/7817_1.csv
```

### Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
tensorflow
joblib
```

## References

- Dataset: [Amazon Product Reviews Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset)
- He et al. (2017) — Neural Collaborative Filtering
- Scikit-learn NearestNeighbors Documentation
- TensorFlow/Keras Embedding Layer Documentation

---

## Author

**Rafael Gallo**
Data Science and Artificial Intelligence
