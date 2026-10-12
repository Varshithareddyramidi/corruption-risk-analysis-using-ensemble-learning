# Procurement risk prediction

The Streamlit app has two prediction paths:

- **Description only:** Sentence-BERT encodes the tender description. A
  logistic-regression classifier trained on the supplied processed CSV predicts
  its `risk_level` category. No tabular features are required for this path.
- **Structured CSV:** the existing LightGBM and FT-Transformer models trained
  on the Albania procurement dataset remain in place. When a description column
  is present, the Sentence-BERT classifier is included in an equal-weight
  ensemble with those two models. The app also calculates the project's
  documented risk score directly and uses its thresholds for the primary
  `Predicted_Risk_Level`; the ensemble output is retained as
  `Model_Ensemble_Prediction` for comparison. CSVs without descriptions keep
  the LightGBM + FT-Transformer ensemble for that comparison column.

The supplied `procurement_realistic_synthetic_60.csv` has a different schema.
The app now maps its bidder count, estimated value, complaints, restrictive
specification flag, and English description to the model's expected features.
This mapping is approximate, and several features used in training are absent
from the file. The CSV has no verified `risk_level` labels, so its predictions
cannot establish model accuracy. Treat the displayed categories as proxy risk
indicators, not confirmed corruption findings.

The upload dashboard accepts common alternate names for descriptions, bidder
counts, values, complaints, restrictive specifications, procedure type, bid
period, repeat winners, amendments, value increases, and award justifications.
The app distinguishes fields mapped into trained features from additional
review signals that the models were not trained on. SHAP explanations are
shown in plain language: Sentence-BERT groups influential word pieces into
short phrases and estimates their class-score effects. For a selected
structured row, LightGBM and FT-Transformer are explained separately; the
FT-Transformer attribution uses repeated feature-order permutations and a
fixed training-mean baseline. When a description is present, its Sentence-BERT
phrase explanation is shown too. The displayed rule category has a separate
list of its contributing signals and points. Other dashboard charts summarize
risk counts, score distributions, model versus rule categories, and additional
review signals.

## Model evaluation and retraining

The current saved tabular artifacts were retrained with HIGH, LOW, and MEDIUM
present in train, validation, and test splits. On the supplied 151-row test
split, LightGBM scored 99.3% accuracy, FT-Transformer 98.7%, and their
two-model soft-voting ensemble 98.7%. The Sentence-BERT classifier uses
validation data to choose between weighted and unweighted logistic regression;
the current selected option is unweighted. Its separate 201-row holdout scored
92.0% accuracy and 87.0% macro-F1. Detailed metrics are saved in
`models/evaluation_metrics.json` and `models/sbert/metadata.json`.

These metrics measure prediction of the project's generated proxy labels. The
labels are computed from risk indicators that are also model features, so the
high tabular scores mainly show that the model learned that scoring formula.
They do not measure real-world corruption detection. The existing split is
row-based and repeated tender descriptions occur across splits; use a
time-based or procuring-entity-grouped evaluation before claiming performance
on future tenders or unseen entities.

To retrain using the provided splits, run:

```powershell
python main_train_ensemble.py --train "C:\Users\varsh\Downloads\data\processed\train.csv" --validation "C:\Users\varsh\Downloads\data\processed\val.csv" --test "C:\Users\varsh\Downloads\data\processed\test.csv"
```

The script refuses to train or evaluate if any split is missing one of the
three classes and saves per-class test metrics alongside the model artifacts.

## Train the Sentence-BERT classifier

The supplied file `procurement_nlp_processed.csv` contains the required
`tender_description` and `risk_level` columns. From this project folder, run:

```powershell
python -m pip install -r requirements.txt
python train_sbert_classifier.py "C:\Users\varsh\Downloads\procurement_nlp_processed.csv"
```

This uses the pretrained multilingual Sentence-BERT encoder with its weights
frozen, then fits a classifier on the embeddings. The model must be available in
the local Hugging Face cache for offline training. The deployed app downloads
the encoder from Hugging Face on its first start if it is not cached. Training saves only the new text classifier,
label encoder, and metadata under `models/sbert/`; the Albania-trained LightGBM
and FT-Transformer artifacts are not retrained or replaced.

## Run the app

```powershell
python -m streamlit run app.py
```

Paste one tender description under **Predict from a description**, or upload a
structured CSV in the sidebar.

## Interpreting the result

The new text classifier learns the processed CSV's existing `risk_level`
categories. Those are rule-generated proxy labels, not independently confirmed
corruption findings. The classifier therefore learns to predict those labels;
the structured-data primary category applies the same score formula directly,
while ML outputs are shown separately so their agreement with that rule is
visible. This training data alone does not establish real-world corruption
accuracy. Treat results as screening estimates for human review, not findings
of wrongdoing. Displayed class percentages are model scores and are not
calibrated as real-world probabilities.
