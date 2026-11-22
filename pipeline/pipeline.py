from kfp import compiler, dsl
from google.cloud import aiplatform
import config

SERVE_IMG = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest"

@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "google-cloud-bigquery", "db-dtypes", "pandas",
        "scikit-learn==1.6.1", "joblib",
    ],
)
def train(
    project: str, table: str, C: float,
    model: dsl.Output[dsl.Model], metrics: dsl.Output[dsl.Metrics],
) -> float:
    import os, joblib
    from google.cloud import bigquery
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline

    df = bigquery.Client(project=project).query(
        f"SELECT review, label, split FROM `{table}`"
    ).to_dataframe()
    tr, te = df[df.split == "train"], df[df.split == "test"]
    m = make_pipeline(
        TfidfVectorizer(max_features=50_000, ngram_range=(1, 2)),
        LogisticRegression(C=C, max_iter=1000),
    ).fit(tr.review, tr.label)
    acc = float(accuracy_score(te.label, m.predict(te.review)))
    metrics.log_metric("accuracy", acc)
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(m, os.path.join(model.path, "model.joblib"))
    return acc

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"],
)
def register(
    project: str, region: str, serve_img: str, model: dsl.Input[dsl.Model]
):
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)
    existing = aiplatform.Model.list(filter='display_name="sentiment-classifier"')
    aiplatform.Model.upload(
        display_name="sentiment-classifier",
        artifact_uri=model.uri,
        serving_container_image_uri=serve_img,
        parent_model=existing[0].resource_name if existing else None,
    )

@dsl.pipeline(name="sentiment-train")
def sentiment_pipeline(
    project: str, table: str, C: float = 1.0, threshold: float = 0.85
):
    t = train(project=project, table=table, C=C)
    with dsl.If(t.outputs["Output"] >= threshold, name="quality-gate"):
        register(
            project=project,
            region="us-central1",
            serve_img=SERVE_IMG,
            model=t.outputs["model"],
        )

if __name__ == "__main__":
    compiler.Compiler().compile(sentiment_pipeline, "pipeline/sentiment.json")
    aiplatform.init(project=config.PROJECT, location=config.REGION)
    aiplatform.PipelineJob(
        display_name="sentiment-train",
        template_path="pipeline/sentiment.json",
        pipeline_root=f"{config.BUCKET}/pipeline_root",
        parameter_values={
            "project": config.PROJECT,
            "table": f"{config.PROJECT}.{config.BQ_DATASET}.reviews",
        },
    ).submit(
        service_account=f"p1-trainer@{config.PROJECT}.iam.gserviceaccount.com"
    )
