import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["GROQ_API_KEY"]
response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {api_key}"},
)
model_ids = [m["id"] for m in response.json()["data"]]
for model_id in sorted(model_ids):
    print(model_id)