# Scores 1,000 test reviews from a file. No endpoint is involved.
# Run from the repo root: python serving/batch.py

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.cloud import aiplatform, bigquery, storage

import config

MODEL = "projects/668878719281/locations/us-central1/models/3539328479566757888@champion"

df = bigquery.Client(project=config.PROJECT).query(
    f"SELECT review FROM `{config.PROJECT}.{config.BQ_DATASET}.reviews` "
    "WHERE split='test' LIMIT 1000"
).to_dataframe()
blob = (
    storage.Client(project=config.PROJECT)
    .bucket(config.BUCKET[5:])
    .blob("batch/input.jsonl")
)
blob.upload_from_string("\n".join(json.dumps(r) for r in df.review))

aiplatform.init(project=config.PROJECT, location=config.REGION)
model = aiplatform.Model(MODEL)
job = model.batch_predict(
    job_display_name="nightly-sentiment",
    gcs_source=f"{config.BUCKET}/batch/input.jsonl",
    gcs_destination_prefix=f"{config.BUCKET}/batch/output",
    instances_format="jsonl",
    machine_type="n1-standard-2",
    sync=True,
)
print(job.state, job.output_info.gcs_output_directory)
