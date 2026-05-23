#!/bin/bash
# Manual smoke test against the deployed Cloud Run service.
# NOT run in CI (needs auth + costs a few paise). Run on demand.
set -e
URL="https://rag-api-906503408814.asia-south1.run.app"
TOKEN=$(gcloud auth print-identity-token)

echo "== health =="
curl -s -H "Authorization: Bearer $TOKEN" "$URL/health"; echo

echo "== ask =="
curl -s -X POST "$URL/ask" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is multi-head attention?","backend":"vertex","k":3}' \
  | python3 -m json.tool
