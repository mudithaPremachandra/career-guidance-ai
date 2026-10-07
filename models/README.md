# Trained models

Produced by `python -m src.train` (run from the repository root).

| File | What it is |
|---|---|
| `random_forest.joblib` | The ML engine's default classifier: the exact model the app trains on startup (80 trees, max depth 10) |
| `decision_tree.joblib` | Single-tree baseline (max depth 10), trained on the same split |
| `metrics.json` | Train/test accuracy, 5-fold cross-validation, per-career precision/recall/F1, confusion matrix, top features, and the library versions used |

Training data: `data/benchmark_dataset.csv` (405 synthetic students, 45 per career). Because it is generated from the
same benchmark matrix the rule and fuzzy engines use, the scores show internal consistency, not accuracy on real students.

The app does not load these files. It retrains the identical model in memory on startup (under a second), so a
different scikit-learn version on the server can never break loading a saved pickle. To load one yourself:

```python
import joblib
model = joblib.load("models/random_forest.joblib")
```
Use the same scikit-learn version listed in `metrics.json`.
