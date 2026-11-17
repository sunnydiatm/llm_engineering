import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key.startswith('sk-') and len(api_key) > 10:
    print("ChatGpt API key looks good so far")
else:
    print("There might be a problem with your API key!")

question = """
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""

system_prompt = """
You are a technical expert instructor/teacher and 
when user is asking/posting any technical question then answer the question in that capacity
"""

openai = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
response = openai.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
          ]
    )


print(response.choices[0].message.content)



