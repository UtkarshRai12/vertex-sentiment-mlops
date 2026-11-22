# Creates the online endpoint, deploys @champion, and scores two reviews.
# A replica bills by the hour from the moment it is up, including idle.
# Run scripts/teardown.py before you stop.
#
# Run from the repo root: python serving/deploy.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.cloud import aiplatform

import config

MODEL = "projects/668878719281/locations/us-central1/models/3539328479566757888@champion"

aiplatform.init(project=config.PROJECT, location=config.REGION)
endpoint = aiplatform.Endpoint.create(
    display_name="sentiment-endpoint",
    enable_request_response_logging=True,
    request_response_logging_sampling_rate=1.0,
    request_response_logging_bq_destination_table=(
        f"bq://{config.PROJECT}.{config.BQ_DATASET}.prediction_logs"
    ),
)
model = aiplatform.Model(MODEL)
model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=2,
    traffic_percentage=100,
)
print(
    endpoint.predict(
        instances=[
            "An absolute masterpiece, I cried twice.",
            "Two hours of my life I will never get back.",
        ]
    ).predictions
)
