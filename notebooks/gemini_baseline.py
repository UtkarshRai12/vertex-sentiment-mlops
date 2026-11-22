# Zero-shot baseline. Score 200 test reviews with Gemini on Vertex.
#
# This number is the bar the later TF-IDF model has to beat, or justify
# being cheaper and faster than. The script also records token use and
# turns it into a list-price cost.
#
# Run from the repo root, with the venv active and gcloud config p1:
#   python notebooks/gemini_baseline.py

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai
from google.cloud import bigquery
from google.genai import types

import config

# gemini-2.5-flash was retired on 2026-10-20. Google's replacement on
# Vertex is gemini-3.5-flash-lite, and this project cannot call it from
# us-central1. The global endpoint serves it. List price there matches
# the old Flash rate: $0.30 / 1M input, $2.50 / 1M output.
# Source: https://cloud.google.com/model-garden/pricing
# Output price includes the reply and any reasoning tokens.
INPUT_USD_PER_MILLION = 0.30
OUTPUT_USD_PER_MILLION = 2.50
MODEL = "gemini-3.5-flash-lite"
MODEL_LOCATION = "global"
SAMPLE_SIZE = 200

# Same 200 rows on every run. LIMIT alone would change the sample.
QUERY = f"""
SELECT review, label
FROM `{config.PROJECT}.{config.BQ_DATASET}.reviews`
WHERE split = 'test'
ORDER BY FARM_FINGERPRINT(review)
LIMIT {SAMPLE_SIZE}
"""


def call_model(client, review):
    """Ask for one word. Retry a few times if Vertex rate-limits the call."""
    prompt = (
        "Is this movie review positive or negative? Reply with one word.\n\n"
        f"{review[:2000]}"
    )
    # thinking_budget=0 turns reasoning off. We asked for one word, and
    # reasoning tokens are billed at the output rate.
    request = dict(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=16,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    delay = 2
    for attempt in range(5):
        try:
            return client.models.generate_content(**request)
        except Exception as exc:
            if attempt == 4:
                raise
            print(f"call failed ({exc}). retrying in {delay}s", flush=True)
            time.sleep(delay)
            delay *= 2


def main():
    client = genai.Client(
        vertexai=True, project=config.PROJECT, location=MODEL_LOCATION
    )
    rows = bigquery.Client(project=config.PROJECT).query(QUERY).to_dataframe()
    if len(rows) != SAMPLE_SIZE:
        raise SystemExit(f"expected {SAMPLE_SIZE} test rows, got {len(rows)}")

    correct = 0
    prompt_tokens = 0
    output_tokens = 0
    thought_tokens = 0

    for i, (review, label) in enumerate(zip(rows.review, rows.label), start=1):
        response = call_model(client, review)
        answer = (response.text or "").strip().lower()
        # label 1 is positive. Any reply containing "positive" counts as positive.
        correct += int(("positive" in answer) == bool(label))

        usage = response.usage_metadata
        prompt_tokens += usage.prompt_token_count or 0
        output_tokens += usage.candidates_token_count or 0
        thought_tokens += usage.thoughts_token_count or 0

        if i % 20 == 0 or i == SAMPLE_SIZE:
            print(f"scored {i}/{SAMPLE_SIZE}", flush=True)

    accuracy = correct / SAMPLE_SIZE
    # Thoughts are billed with the reply, at the output rate.
    billed_output_tokens = output_tokens + thought_tokens
    input_usd = prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION
    output_usd = billed_output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    total_usd = input_usd + output_usd

    summary = (
        f"Gemini zero-shot accuracy on {SAMPLE_SIZE} reviews: {accuracy:.3f}\n"
        f"prompt tokens: {prompt_tokens}\n"
        f"output tokens: {output_tokens}\n"
        f"thought tokens: {thought_tokens}\n"
        f"estimated cost USD: {total_usd:.6f}\n"
    )
    print(summary, end="")

    # Kept next to the script so Phase 8 can quote the same numbers.
    result_path = Path(__file__).resolve().parent / "gemini_baseline_result.md"
    result_path.write_text(
        "\n".join(
            [
                "# Gemini zero-shot baseline",
                "",
                f"- Model: `{MODEL}` on Vertex, location `{MODEL_LOCATION}`",
                "- `gemini-2.5-flash` was retired on 2026-10-20. This is the current Flash replacement, and it is not served from `us-central1`.",
                f"- Sample: {SAMPLE_SIZE} rows from `sentiment.reviews` where `split = 'test'`, ordered by `FARM_FINGERPRINT(review)`",
                "- Thinking: off (`thinking_budget=0`). Temperature: 0.",
                f"- Accuracy: **{accuracy:.3f}** ({correct}/{SAMPLE_SIZE})",
                f"- Prompt tokens: {prompt_tokens}",
                f"- Output tokens: {output_tokens}",
                f"- Thought tokens: {thought_tokens}",
                f"- List price: ${INPUT_USD_PER_MILLION:.2f} / 1M input tokens, ${OUTPUT_USD_PER_MILLION:.2f} / 1M output tokens (reply and reasoning)",
                f"- Estimated cost: **${total_usd:.6f}** (input ${input_usd:.6f} + output ${output_usd:.6f})",
                "- This is the published Vertex list price in USD, not the invoice. The billing account is INR, so the bill converts this amount.",
                "- Price source: https://cloud.google.com/model-garden/pricing (Gemini 3.5 Flash-Lite, global, standard, text, prompts under 200k tokens).",
                "",
            ]
        )
    )
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
