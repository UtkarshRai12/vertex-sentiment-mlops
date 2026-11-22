# Gemini zero-shot baseline

- Model: `gemini-3.5-flash-lite` on Vertex, location `global`
- `gemini-2.5-flash` was retired on 2026-10-20. This is the current Flash replacement, and it is not served from `us-central1`.
- Sample: 200 rows from `sentiment.reviews` where `split = 'test'`, ordered by `FARM_FINGERPRINT(review)`
- Thinking: off (`thinking_budget=0`). Temperature: 0.
- Accuracy: **0.965** (193/200)
- Prompt tokens: 49760
- Output tokens: 200
- Thought tokens: 0
- List price: $0.30 / 1M input tokens, $2.50 / 1M output tokens (reply and reasoning)
- Estimated cost: **$0.015428** (input $0.014928 + output $0.000500)
- This is the published Vertex list price in USD, not the invoice. The billing account is INR, so the bill converts this amount.
- Price source: https://cloud.google.com/model-garden/pricing (Gemini 3.5 Flash-Lite, global, standard, text, prompts under 200k tokens).
