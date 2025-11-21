
# DA5401 Data Challenge: Semantic Similarity in AI Evaluation

---

**Student:** S Shriprasad  
**Roll No:** DA25E054  
**Course:** DA5401 (Data Science)

---

## 📌 Project Overview

Evaluating Generative AI is subjective and difficult to scale. This project aims to build an **Automated Metric Learning Model** that predicts a "Fitness Score" (0-10) representing how well an AI's response adheres to a specific evaluation metric definition.

**The Core Challenge:** The dataset represents an **Anomaly Detection** problem disguised as regression. Over **90% of the data has a score > 8.0**, meaning "bad" responses are statistically rare. Standard models trained on this data simply learn to predict the mean (~9.1), failing to capture actual metric alignment.

---

## 🚀 Key Technical Innovations

*   **Strategic Pivot:** Moved from unstable Deep Learning architectures (Siamese Networks) to robust Gradient Boosting (XGBoost) with engineered features.
*   **Dimensionality Reduction:** Utilized **PCA** to compress 768-dimensional MPNet embeddings into 50 informative components, preventing the "Curse of Dimensionality."
*   **Synthetic Negative Sampling:** Addressed extreme class imbalance by creating "Synthetic Negatives" (Metric Shuffling), forcing the model to learn what a mismatch looks like.
*   **Multilingual Support:** Leveraged `paraphrase-multilingual-mpnet-base-v2` to handle Hindi, Tamil, and English prompts seamlessly.

---

## 📊 Exploratory Data Analysis (EDA)

My analysis focused on the geometry of the data and the distribution of the target variable:

1.  **Metric Clustering (PCA):** Visualizing metric embeddings in 2D space revealed distinct clusters (e.g., "Safety" metrics vs. "Formatting" metrics), confirming that the semantic vectors contain discriminative signals.
2.  **The "Cliff" Distribution:** The CDF plot confirmed that less than 10% of the data falls below a score of 8.
3.  **Length vs. Quality:** Correlation analysis showed **zero correlation (0.017)** between text length and score. Verbosity does not equal quality.

---

## 🧪 Methodology

### 1. The Failed Experiment: Siamese Networks
I initially attempted to solve this using a **Siamese LSTM Network**, commonly used for semantic similarity.
*   **Architecture:** Shared LSTM Encoders + Manhattan Distance.
*   **Outcome:** Failed (RMSE plateaued at ~3.0).
*   **Why?** Contrastive loss requires balanced Positive/Negative pairs. The lack of "negative" examples caused the network to collapse and predict "Similar" for everything.

### 2. The Solution: Feature Engineering + XGBoost
To overcome the stability issues of Deep Learning on small data, I treated the similarity scores as engineered features:

*   **Step A: Synthetic Data Augmentation:**
    I generated synthetic data where `Score = random(0, 3)` by pairing high-quality responses with *random, unrelated* metric definitions. This provided the "negative signal" the model desperately needed.

*   **Step B: Feature Construction:**
    Instead of raw text, I fed XGBoost a dense feature matrix:
    *   **Cosine Similarity:** $\cos(\vec{u}_{metric}, \vec{v}_{response})$ (Computed on Raw Embeddings).
    *   **Euclidean Distance:** Magnitude of difference.
    *   **PCA Components:** The top 50 principal components of the Response and Metric embeddings to capture latent semantic topics.

*   **Step C: XGBoost Regressor:**
    Trained with `reg:squarederror`, `max_depth=8`, and `learning_rate=0.05`.

---

## 🛠️ Installation & Usage

### Requirements
```txt
pandas
numpy
scikit-learn
sentence-transformers
xgboost
matplotlib
seaborn
langdetect
```

### Running the Project
1.  Place `train_data.json`, `test_data.json`, and `metric_names.json` in the root directory.
2.  Run the notebook cells sequentially.
    *   **Note:** The PCA reduction step ensures the feature space is manageable for XGBoost.
    *   **Note:** `trained_pca_reducers` are saved during training and reapplied to the Test set to ensure consistent feature spaces.

---

## 📈 Results

The XGBoost model, aided by synthetic negatives, successfully learned to penalize mismatches.

| Feature | Importance Rank | Interpretation |
| :--- | :--- | :--- |
| **Cosine Similarity (Response-Metric)** | 1 | Primary driver of the score. |
| **Cosine Similarity (Prompt-Metric)** | 2 | High alignment implies the metric is relevant to the user's request. |
| **Response Length** | Low | Confirms EDA; length is not a proxy for quality. |

---

## 📜 Conclusion

This project demonstrates that **Domain-Specific Feature Engineering** often outperforms complex end-to-end Deep Learning when data is scarce or highly skewed. By manually computing similarities and reducing dimensions via PCA, I provided the XGBoost model with a distilled, high-signal representation of the problem, solving the "perfect score" bias effectively.
