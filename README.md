# Prerequisites

## Clone the repo
```bash
git clone https://github.com/yoshikado/ai-mbo-apac-fe-bravo.git
cd ai-mbo-apac-fe-bravo
```
## Set virtual env
```bash
sudo apt update
sudo apt install python3.12-venv -y
python3 -m venv venv
source venv/bin/activate
```
## Install python libraries
```bash
pip install fastapi uvicorn openai python-dotenv pydantic chromadb httpx pypdf
```
## Update Cloudflare account ID and API token
Update `.env` file with the Cloudflare's accound ID and API token.

# Run the service
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

# Ingest data
```bash
url="http://10.95.36.171:8000"
curl -X 'POST' \
  "${url}/ingest/url" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "url": "https://ubuntu.com/legal/ubuntu-pro-description",
  "component": "Ubuntu Pro",
  "version": "2026-06-29-v1"
}'
```

# Query question
```bash
url="http://10.95.36.171:8000"
curl --no-progress-meter -X 'POST' \
  "${url}/query" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What is the specific scope difference between standard Ubuntu Pro and Ubuntu Pro Infra-only subscriptions regarding Universe repository packages?"
}' | jq .

{
  "interaction_id": "dbac113f-7db2-4737-ab98-9b72b96d9636",
  "answer": "The specific scope difference regarding the Universe repository is as follows:\n\n*   **Ubuntu Pro (+ Support):** Includes all packages in the **Ubuntu Universe** repository, starting with 18.04 LTS and onwards (Section 8.3.1).\n*   **Ubuntu Pro (Infra-only):** Does not include the Universe repository; its scope for packages is limited to **all packages in Ubuntu Main** (Section 8.2.8). The inclusion of Universe packages is explicitly defined as an addition to the infra-only support (Section 8.3).",
  "sources": [
    {
      "component": "Ubuntu Pro",
      "source": "ubuntu-pro-description",
      "url": "https://ubuntu.com/legal/ubuntu-pro-description"
    }
  ]
}
```