---
title: Legal Assistant
emoji: 🐨
colorFrom: indigo
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: apache-2.0
short_description: Legal Assistant using vectara-agentic
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

To run locally:
Set the environment variables:
```
VECTARA_CUSTOMER_ID
VECTARA_API_KEY
VECTARA_CORPUS_ID
QUERY_EXAMPLES = "Tell me about Konrad v State"
PHOENIX_ENDPOINT = "http://0.0.0.0:4317"
OPENAI_API_KEY =
```

Run a Phoenix instance:
```
pip install arize-phoenix
python -m phoenix.server.main serve
```

Install the requirements:
```
pip install -r requirements.txt
```

Run the app:
```
streamlit run app.py
```