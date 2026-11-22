# Undeploys every endpoint so idle replicas stop billing.
# Run from the repo root: python scripts/teardown.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.cloud import aiplatform

import config

aiplatform.init(project=config.PROJECT, location=config.REGION)
for ep in aiplatform.Endpoint.list():
    ep.undeploy_all()
    print("undeployed", ep.display_name)
