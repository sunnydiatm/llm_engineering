import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import sqlite3

load_dotenv(override=True)
chatgpt_api_key = os.getenv('OPENAI_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

if gemini_api_key and gemini_api_key.startswith('AQ') and len(gemini_api_key) > 10:
    print("GEMINI API key looks good so far")
elif chatgpt_api_key and chatgpt_api_key.startswith('sk-') and len(chatgpt_api_key) > 10:
    print("ChatGPT API key looks good so far")
else:
    print("There might be a problem with your API key!")

# Use for Llama Models
#LLAMA_BASE_URL = "http://localhost:11434/v1"
#MODEL = "llama3.2"
#openai = OpenAI(api_key=api_key,base_url=LLAMA_BASE_URL)

#Use for Chat GPT Models
#MODEL = "gpt-6-luna"
#openai = OpenAI()

#Use for Gemini GPT Models
MODEL = "gemini-3.5-flash"
openai = OpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta")

# DB stuff

DB = "prices.db"

with sqlite3.connect(DB) as conn:
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)')
    conn.commit()


def get_ticket_price(city):
    print(f"DATABASE TOOL CALLED: Getting price for {city}", flush=True)
    if (city == "None"):
        return
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT price FROM prices WHERE city = ?', (city.lower(),))
        result = cursor.fetchone()
        return f"Ticket price to {city} is ${result[0]}" if result else "No price data available for this city"


def set_ticket_price(city, price):
    print(f"DATABASE TOOL CALLED: Setting price for {city} and {price}", flush=True)
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO prices (city, price) VALUES (?, ?) ON CONFLICT(city) DO UPDATE SET price = ?',(city.lower(), price, price))
        conn.commit()

system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""


def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, reasoning_effort="none", messages=messages, tools=tools)

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses = handle_tool_calls(message)
        messages.append(message)
        messages.extend(responses)
        response = openai.chat.completions.create(model=MODEL, reasoning_effort="none", messages=messages, tools=tools)

    return response.choices[0].message.content



price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}


tools = [{"type": "function", "function": price_function}]

def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses

# Insert some ticket prices
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

for city, price in ticket_prices.items():
    set_ticket_price(city, price)

#run with Gradio
gr.ChatInterface(fn=chat).launch()