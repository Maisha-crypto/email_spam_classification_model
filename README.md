# Email Spam Classification with Machine Learning

This project explores automatic email spam classification using natural language processing (NLP) and supervised machine learning. The analysis is implemented in [`text_classification_model.ipynb`](./text_classification_model.ipynb).

## What the project does

The notebook:

1. Loads the email dataset from [`dataset/emails.csv`](./dataset/emails.csv).
2. Checks the dataset structure, missing values, and duplicate messages.
3. Removes duplicate email text, reducing the dataset from 5,728 to 5,695 rows.
4. Explores the class distribution and basic email characteristics, including character, word, and sentence counts.
5. Preprocesses email text by lowercasing, tokenizing, removing non-alphanumeric tokens and stop words, and applying Porter stemming.
6. Creates TF-IDF features with a maximum of 3,000 features.
7. Splits the data into training and test sets using an 80/20 split with `random_state=42`.
8. Trains and evaluates several classification algorithms using accuracy, precision, recall, F1 score, classification reports, and confusion matrices.

The dataset contains two columns:

- `text`: the email content.
- `spam`: the encoded target label. In the supplied data, spam examples use `1` and non-spam examples use `0`.

After duplicate removal, the notebook contains 1,368 spam messages and 4,327 non-spam messages. This class imbalance is visualized during the exploratory analysis and should be considered when interpreting accuracy.

## Models compared

The notebook evaluates:

- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Support Vector Classifier
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Bagging
- Extra Trees
- XGBoost

In the saved notebook results, `ExtraTreesClassifier` achieved the strongest overall result among the tested models:

| Model | Accuracy | Precision | Recall | F1 score |
| --- | ---: | ---: | ---: | ---: |
| ExtraTreesClassifier | 0.9824 | 1.0000 | 0.9324 | 0.9650 |
| MultinomialNB | 0.9807 | 0.9858 | 0.9392 | 0.9619 |
| LogisticRegression | 0.9772 | 0.9655 | 0.9459 | 0.9556 |

These figures are the results saved in the notebook and may vary slightly if the notebook or dependencies are changed.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- Jupyter Notebook or JupyterLab, or an editor such as VS Code with the Jupyter extension

The Python dependencies are declared in [`pyproject.toml`](./pyproject.toml) and locked in [`uv.lock`](./uv.lock). They include pandas, NumPy, scikit-learn, NLTK, XGBoost, Matplotlib, Seaborn, and WordCloud.

## Installation

From the project root, create the environment and install the locked dependencies:

```powershell
uv sync
```

The notebook downloads the NLTK resources `punkt`, `punkt_tab`, and `stopwords` the first time it runs. An internet connection may be required for this step.

## Running the notebook

Start JupyterLab with the project environment and open the notebook:

```powershell
uv run --with jupyter jupyter lab
```

Then open [`text_classification_model.ipynb`](./text_classification_model.ipynb) and run the cells from top to bottom. Run Jupyter from the project root so the relative dataset path resolves correctly.

Alternatively, in VS Code:

1. Open the project folder.
2. Open [`text_classification_model.ipynb`](./text_classification_model.ipynb).
3. Select the Python 3.12 environment created by `uv sync` as the notebook kernel.
4. Run all cells in order.

## Running the package command

The package also exposes a command through `pyproject.toml`:

```powershell
uv run machine-learning-spam-classification-model
```

At present, this command runs the placeholder `main()` function and prints a greeting. The trained model and evaluation workflow currently live in the notebook.

## Project structure

```text
.
├── dataset/
│   └── emails.csv
├── src/
│   └── machine_learning_spam_classification_model/
│       └── __init__.py
├── text_classification_model.ipynb
├── pyproject.toml
├── uv.lock
└── README.md
```

## Limitations and possible improvements

- The model is trained and evaluated in the notebook; no serialized model or prediction API is currently included.
- The classes are imbalanced, so precision, recall, and F1 score should be considered alongside accuracy.
- The notebook imports `kagglehub`, but the current workflow reads the local CSV file.
- The notebook uses a Windows-style relative dataset path. On non-Windows systems, change it to `dataset/emails.csv` before running.
- Future improvements could include stratified splitting, cross-validation, hyperparameter tuning, class balancing, and exporting the selected model and vectorizer for reuse.

## License

No license file is currently included in the repository.
