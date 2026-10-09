# Machine Learning Concepts

Implementations of common machine learning algorithms in NumPy and PyTorch, written for educational purposes. Many notebooks implement the algorithm from scratch first and then compare against the library version. All notebooks are meant to run top to bottom.

## Topics covered

### Regression (`notebooks/regression/`)
- `Linear_Regression.ipynb` - linear regression from scratch with NumPy (gradient descent, L2 regularization, closed form)
- `Linear and Non-Linear Regression Pytorch.ipynb` - linear vs nonlinear regression in PyTorch, including why stacked linear layers stay linear
- `Logistic_Regression.ipynb` - multiclass logistic regression from scratch with NumPy on the Iris dataset
- `Logistic_Regression_SimpleNN_Pytorch.ipynb` - logistic regression, MLPs, overfitting, and regularization (L1, L2, dropout) in PyTorch on the moons dataset

### Trees (`notebooks/trees/`)
- `Decision_tree.ipynb` - information gain and Gini impurity from scratch with NumPy
- `decision_tree-random-forest-pytorch.ipynb` - CART decision tree and random forest from scratch in PyTorch
- `gradient_boosting-pytorch.ipynb` - gradient boosting from scratch in PyTorch (sequential residual fitting)

### Support Vector Machines (`notebooks/svm/`)
- `svm-pytorch.ipynb` - binary classifier trained with the hinge loss in PyTorch

### Naive Bayes (`notebooks/naive-bayes/`)
- `Naive Bayes.ipynb` - multinomial Naive Bayes from scratch with NumPy on the 20 Newsgroups dataset (downloads the dataset on first run)

### K-Nearest Neighbors (`notebooks/knn/`)
- `k-nearest-pytorch.ipynb` - KNN from scratch in PyTorch with decision boundary visualization

### K-Means (`notebooks/kmeans/`)
- `k-means_numpy.ipynb` - k-means from scratch with NumPy
- `k-means_pytorch.ipynb` - k-means from scratch in PyTorch

### Dimensionality Reduction (`notebooks/dim-reduce/`)
- `pca.ipynb` - PCA with scikit-learn and from scratch with NumPy on the Olivetti faces dataset (downloads the dataset on first run)
- `tsne.ipynb` - t-SNE embedding of the digits dataset for visualization

### Neural Networks (`notebooks/neural/`)
- `backprop_pytorch.ipynb` - a fully connected network with hand-written backpropagation on PyTorch tensors
- `CNN_pytorch.ipynb` - a small convolutional network on the digits dataset
- `RNN_pytorch.ipynb` - a character-level LSTM trained on a built-in text sample, with text sampling

### Attention (`notebooks/attention/`)
- `Multihead_attention.ipynb` - multi-head self-attention layer in PyTorch
- `attention_heatmap.ipynb` - visualizing self-attention weights as heatmaps

### Deep Learning Helpers (`notebooks/dl-helpers/`)
- `BatchNorm.ipynb` - batch normalization, layer normalization, and dropout from scratch with NumPy, plus a tiny manual gradient descent demo

### Quantization (`notebooks/quantization/`)
- `quantization.ipynb` - dynamic quantization, static post-training quantization (with layer fusion and calibration), and quantization-aware training on MNIST (downloads MNIST on first run)

### Evaluation Metrics (`notebooks/metrics/`)
- `metrics.ipynb` - confusion matrix, precision/recall/F1, CER/WER, perplexity, cross-entropy, cosine similarity, BLEU, and ROUGE-L from scratch

### Applied Tasks (`notebooks/tasks/`)
- `Pytorch-Classification.ipynb` - text classification on AG News with an embedding-bag model (needs `torchtext` and downloads AG News on first run)
- `Recommendation.ipynb` - item-based collaborative filtering, content-based filtering, and neural collaborative filtering on MovieLens 100k (downloads the dataset on first run)

### Coding Interview (`notebooks/coding_interview/`, `notebooks/trie/`)
- `backtracking/N-queens.ipynb` - the N-Queens problem solved with backtracking, with step-by-step board visualization
- `trie/FST.ipynb` - a trie with greedy decoding, beam search, and CTC beam search constrained to valid words

## How to run

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
   `torchtext` is only needed for the AG News classification notebook. If you skip that notebook, you can skip `torchtext`.

2. Launch Jupyter from the repo root:
   ```
   jupyter notebook
   ```
   then open any notebook under `notebooks/` and run all cells.

Several notebooks download public datasets on first run (20 Newsgroups, Olivetti faces, MNIST, MovieLens 100k, AG News), so the first execution needs internet access. The notebooks were developed on CPU; no GPU is required.

## What you learn

Each notebook follows the same pattern: build intuition with a small example, implement the core algorithm from scratch in NumPy or PyTorch, visualize what the model learned (decision boundaries, embeddings, attention weights), and where it helps, compare the from-scratch version against the library implementation. The goal is to understand how the algorithms work under the hood, not to provide production code.
