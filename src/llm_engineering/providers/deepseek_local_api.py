from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

question = """
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""

system_prompt = """
You are a technical expert instructor/teacher and 
when user is asking/posting any technical question then answer the question in that capacity
"""
BASE_URL = "http://localhost:11434/v1"

openai = OpenAI(api_key="api_key",base_url=BASE_URL)
response = openai.chat.completions.create(
        model="deepseek-r1:1.5b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
          ]
    )

print(response.choices[0].message.content)