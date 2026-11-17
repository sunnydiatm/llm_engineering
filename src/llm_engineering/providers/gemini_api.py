# import os
# from google import genai
# from dotenv import load_dotenv
#
# load_dotenv()
#
# api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
#
# if not api_key:
#     raise ValueError("API Key not found! Check your .env file.")
#
# # Initialize the client
# client = genai.Client(api_key=api_key)
#
# # Use standard content generation instead of interactions.create
# response = client.providers.generate_content(
#     model="gemini-3.5-flash",
#     contents="Explain how AI works in a few words"
# )
#
# print(response.text)


import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv('GEMINI_API_KEY')

if api_key and api_key.startswith('AQ') and len(api_key) > 10:
    print("GEMINI API key looks good so far")
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

openai = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta")
response = openai.chat.completions.create(
        model="gemini-3.5-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
          ]
    )


print(response.choices[0].message.content)









# import os
# import ssl
# import urllib.request
# import json
# from dotenv import load_dotenv
#
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
#
# ctx = ssl.create_default_context()
# ctx.check_hostname = False
# ctx.verify_mode = ssl.CERT_NONE
#
# # Using gemini-3.5-flash
# url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
# payload = {
#     "contents": [{
#         "parts": [{"text": "Say hello!"}]
#     }]
# }
#
# data = json.dumps(payload).encode("utf-8")
# req = urllib.request.Request(
#     url,
#     data=data,
#     headers={"Content-Type": "application/json"},
#     method="POST"
# )
#
# try:
#     print("Sending request to Gemini 3.5 Flash...")
#     with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
#         result = json.loads(response.read().decode("utf-8"))
#         output_text = result["candidates"][0]["content"]["parts"][0]["text"]
#         print("\nSUCCESS! Response:")
#         print(output_text)
# except Exception as e:
#     print(f"Error: {e}")
#
