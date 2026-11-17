import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv('LLAMA_API_KEY')

if api_key and api_key.startswith('llam-') and len(api_key) > 10:
    print("LLama API key looks good so far")
else:
    print("There might be a problem with your API key? Please visit the troubleshooting notebook!")

question = """
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""

system_prompt = """
You are a technical expert instructor/teacher and 
when user is asking/posting any techincal question then answer the question in that capacity
"""
LLAMA_BASE_URL = "http://localhost:11434/v1"

openai = OpenAI(api_key=api_key,base_url=LLAMA_BASE_URL)
response = openai.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
          ]
    )

print(response.choices[0].message.content)