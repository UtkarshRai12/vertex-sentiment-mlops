# Runs on Vertex, not on the laptop. Reads the reviews table, fits
# TF-IDF plus logistic regression, logs the run, and saves model.joblib
# where the serving container expects it.

import argparse
import os
import time

import joblib
from google.cloud import aiplatform, bigquery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

p = argparse.ArgumentParser()
p.add_argument("--project")
p.add_argument("--table")
p.add_argument("--C", type=float, default=1.0)
p.add_argument("--max_features", type=int, default=50_000)
args = p.parse_args()

df = bigquery.Client(project=args.project).query(
    f"SELECT review, label, split FROM `{args.table}`"
).to_dataframe()
train, test = df[df.split == "train"], df[df.split == "test"]

model = make_pipeline(
    TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2)),
    LogisticRegression(C=args.C, max_iter=1000),
)
model.fit(train.review, train.label)
pred = model.predict(test.review)
metrics = {
    "accuracy": float(accuracy_score(test.label, pred)),
    "f1": float(f1_score(test.label, pred)),
}
print(metrics)

aiplatform.init(project=args.project, location="us-central1", experiment="sentiment-exp")
aiplatform.start_run(f"run-{int(time.time())}")
aiplatform.log_params({"C": args.C, "max_features": args.max_features})
aiplatform.log_metrics(metrics)
aiplatform.end_run()

# Vertex mounts the AIP_MODEL_DIR bucket at /gcs.
out = os.environ["AIP_MODEL_DIR"].replace("gs://", "/gcs/")
os.makedirs(out, exist_ok=True)
joblib.dump(model, os.path.join(out, "model.joblib"))
