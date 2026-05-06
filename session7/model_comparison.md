# Model comparison

Sorted by PR-AUC descending. Operating threshold per model is the F1-maximizing threshold unless the caller supplied one explicitly. The `*@Recall=0.80` columns report a fixed-recall operating point for like-for-like comparison.

| Model | PR-AUC | ROC-AUC | Precision | Recall | Threshold | Precision@Recall=0.80 | Recall@Recall=0.80 |
|---|---|---|---|---|---|---|---|
| logistic (l1_balanced) | 0.3712 | 0.8553 | 0.4509 | 0.3298 | 1.0000 | 0.1341 | 0.8002 |
| random_forest (depth=None_leaf=20) | 0.2724 | 0.8995 | 0.2653 | 0.6045 | 0.3604 | 0.2156 | 0.8002 |
| knn (k=25) | 0.2183 | 0.7845 | 0.2166 | 0.5869 | 0.0774 | 0.0492 | 1.0000 |
