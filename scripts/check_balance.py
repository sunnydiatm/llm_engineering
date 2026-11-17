import requests
import os
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("OPENROUTER_API_KEY")

response = requests.get(
    "https://openrouter.ai/api/v1/credits",
    headers={"Authorization": f"Bearer {api_key}"}
)

print("Status:", response.status_code)
print(response.json())