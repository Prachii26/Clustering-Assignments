# 📊 Clustering Algorithms Playground

This repository contains a collection of Jupyter notebooks that
demonstrate a wide range of **clustering** and **anomaly detection**
methods across different data types---tabular, text, images, audio, and
time-series.

## 📁 Repository Structure

    .
    ├── Notebooks/
    │   ├── K-Means from scratch
    │   ├── Hierarchical clustering
    │   ├── GMM clustering
    │   ├── DBSCAN with PyCaret
    │   ├── Anomaly detection with PyOD
    │   ├── Time-series clustering
    │   ├── Document clustering with LLM
    │   ├── Image clustering with ImageBind
    │   └── Audio clustering with ImageBind
    └── README.md

## 🧰 Installation & Setup

### Clone the repository

``` bash
git clone https://github.com/Prachii26/Clustering-Assignments.git
cd Notebooks
```

### Install dependencies

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy \
            tslearn yfinance pyod sentence-transformers umap-learn \
            torch torchvision torchaudio pillow requests librosa soundfile
pip install git+https://github.com/facebookresearch/ImageBind.git
```

## 📝 Notebook Descriptions

### K-Means from Scratch

Manual implementation of K-Means with evaluation metrics and cluster
summaries.

### Hierarchical Clustering

Agglomerative clustering with dendrograms and climate-based dataset.

### GMM Clustering

Gaussian Mixture Models with BIC/AIC model selection.

### DBSCAN with PyCaret

Density-based clustering and outlier detection.

### Anomaly Detection with PyOD

Evaluates multiple anomaly models on synthetic network data.

### Time-Series Clustering

DTW-based clustering of stock price time-series.

### Document Clustering with LLM Embeddings

Sentence-transformer embeddings, UMAP visualization, multiple
algorithms.

### Image Clustering with ImageBind

ImageBind embeddings clustered with UMAP visualization.

### Audio Clustering with ImageBind

Synthetic ImageBind-like audio embeddings clustered and analyzed.

## ▶️ How to Run

1.  Open the `Notebooks/` folder.
2.  Select a notebook.
3.  Run the cells sequentially.

## 🎯 Learning Outcomes

-   Understand major clustering algorithms.
-   Evaluate cluster quality.
-   Work with multimodal data (text, images, audio, time-series).
-   Use modern embedding models.

## 📜 Author
Prachi Gupta
SJSU ID: 019106594
