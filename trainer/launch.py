# Submits the training job. The laptop does not fit the model.
# Run from the repo root: python -m trainer.launch --C 1.0
#
# sklearn-cpu.1-0 is past its end of availability. 1-6 is the matched
# train and serve pair that is still available.

import argparse

from google.cloud import aiplatform

import config

TRAIN_IMG = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest"
SERVE_IMG = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest"
SA = f"p1-trainer@{config.PROJECT}.iam.gserviceaccount.com"

p = argparse.ArgumentParser()
p.add_argument("--C", default="1.0")
p.add_argument("--parent_model")
a = p.parse_args()

aiplatform.init(
    project=config.PROJECT, location=config.REGION, staging_bucket=config.BUCKET
)
job = aiplatform.CustomTrainingJob(
    display_name="sentiment-tfidf",
    script_path="trainer/task.py",
    container_uri=TRAIN_IMG,
    requirements=["google-cloud-bigquery", "db-dtypes", "google-cloud-aiplatform"],
    model_serving_container_image_uri=SERVE_IMG,
)
model = job.run(
    args=[
        "--project", config.PROJECT,
        "--table", f"{config.PROJECT}.{config.BQ_DATASET}.reviews",
        "--C", a.C,
    ],
    machine_type="n1-standard-4",
    service_account=SA,
    model_display_name="sentiment-classifier",
    parent_model=a.parent_model,
)
print("model:", model.resource_name, "version:", model.version_id)
