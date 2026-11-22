# Lists model versions, then puts the champion alias on the better run.
# Run from the repo root: python -m trainer.promote
#
# Version 1 is C=1.0, accuracy 0.8668. Version 2 is C=0.3, accuracy 0.8470.
# Champion is version 1. Change WINNER before running to move it.

from google.cloud import aiplatform

import config

MODEL = "projects/668878719281/locations/us-central1/models/3539328479566757888"
WINNER = "1"

aiplatform.init(project=config.PROJECT, location=config.REGION)
registry = aiplatform.models.ModelRegistry(model=MODEL)
for v in registry.list_versions():
    print(v.version_id, v.version_aliases)
registry.add_version_aliases(["champion"], version=WINNER)
print(f"champion -> version {WINNER}")
