import json
import os

from click import argument
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import base64
from io import BytesIO
from PIL import Image

from llm_tools import get_ticket_price

# How can I become AI Engineer, give me a path and all resources which can help me become one


load_dotenv(override=True)
gemini_api_key = os.getenv('GEMINI_API_KEY')
chatgpt_api_key = os.getenv('OPENAI_API_KEY')
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
llama_api_key = os.getenv('LLAMA_API_KEY')

if gemini_api_key and gemini_api_key.startswith('AQ') and len(gemini_api_key) > 10:
    print("GEMINI API key looks good so far")
elif groq_api_key and groq_api_key.startswith('gsk_') and len(groq_api_key) > 10:
    print("GROQ API key looks good so far")
elif chatgpt_api_key and chatgpt_api_key.startswith('sk-') and len(chatgpt_api_key) > 10:
    print("ChatGPT API key looks good so far")
elif openrouter_api_key and openrouter_api_key.startswith('sk-') and len(openrouter_api_key) > 10:
    print("OpenRouter API key looks good so far")
elif llama_api_key and len(llama_api_key) > 10:
    print("Llama API key looks good so far")
else:
    print("There might be a problem with your API key!")

def chat_gemini(message, history):
    print("Conversation with Gemini LLM started...")
    openai = OpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta")
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": "You are a helpful assistant"}] + history + [{"role": "user", "content": message}]
    stream = openai.chat.completions.create(model="gemini-3.5-flash", reasoning_effort="none", messages=messages, stream=True)
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

def chat_gpt(message, history):
    print("Conversation with ChatGPT LLM started...")
    openai = OpenAI(api_key=chatgpt_api_key, base_url="https://api.openai.com/v1")
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": "You are a helpful assistant"}] + history + [{"role": "user", "content": message}]
    stream = openai.chat.completions.create(model="gpt-6-luna", reasoning_effort="none", messages=messages, stream=True)
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

def chat_groq(message, history):
    print("Conversation with GROQ LLM started...")
    openai = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": "You are a helpful assistant"}] + history + [{"role": "user", "content": message}]
    stream = openai.chat.completions.create(model="openai/gpt-oss-20b", reasoning_effort="low", messages=messages, stream=True)
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

def chat_openrouter(message, history):
    print("Conversation with OpenRouter LLM started...")
    openai = OpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1")
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": "You are a helpful assistant"}] + history + [{"role": "user", "content": message}]
    stream = openai.chat.completions.create(model="openrouter/free", messages=messages, stream=True)
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

def chat_llama(message, history):
    print("Conversation with Llama LLM started...")
    openai = OpenAI(api_key=llama_api_key, base_url="http://localhost:11434/v1")
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": "You are a helpful assistant"}] + history + [{"role": "user", "content": message}]
    stream = openai.chat.completions.create(model="llama3.2", reasoning_effort="none", messages=messages, stream=True)
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response


def artist(city):
    print("Generating image for given city...")
    openai = OpenAI(api_key=chatgpt_api_key, base_url="https://api.openai.com/v1")
    image_response = openai.images.generate(model="gpt-image-1-mini",
            prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
            size="1024x1024",
            n=1,
        )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


def talker(message):
    print("Calling LLM with Voice...")
    openai = OpenAI(api_key=chatgpt_api_key, base_url="https://api.openai.com/v1")
    response = openai.audio.speech.create(model="gpt-4o-mini-tts",
      voice="onyx",    # Also, try replacing onyx with alloy or coral
      input=message
    )
    return response.content

system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

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

def chat(history):
    print("Conversation with Chat LLM started...")
    openai = OpenAI(api_key=gemini_api_key, base_url="https://generativelanguage.googleapis.com/v1beta")
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model="gemini-3.5-flash",
                                            reasoning_effort="none",
                                            messages=messages,
                                            tools=tools)
    cities = []
    image = None

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses, cities = handle_tool_calls_and_return_cities(message)

    reply = response.choices[0].message.content
    history += [{"role": "assistant", "content": reply}]

    print("Going to talker..")
    voice = talker(reply)

    if cities:
        print("Going to generated image..")
        image = artist(cities[0])

    return history, voice, image



def handle_tool_calls_and_return_cities(message):
    responses = []
    cities = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get("destination_city")
            cities.append(city)
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id,
            })
    return responses, cities

# Callbacks (along with the chat() function above)
def put_message_in_chatbot(message, history):
    return "", history +  [{"role":"user", "content":message}]

# UI definition
# with gr.Blocks() as ui:
#     with gr.Row():
#         chatbot = gr.Chatbot(height=500)
#         image_output = gr.Image(height=500, interactive=False)
#     with gr.Row():
#         audio_output = gr.Audio(autoplay=True)
#     with gr.Row():
#         message = gr.Textbox(label="Chat with our AI Assistant:")

with gr.Blocks(title="FlightAI Assistant") as ui:

    gr.Markdown("# ✈️ FlightAI – Travel Assistant")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat with FlightAI",
                height=450
            )

        with gr.Column(scale=1):
            image_output = gr.Image(
                label="Destination Image",
                height=450,
                interactive=False
            )

    with gr.Row():
        audio_output = gr.Audio(
            label="AI Voice Response",
            autoplay=True
        )

    with gr.Row():
        message = gr.Textbox(
            label="Chat with our AI Assistant",
            placeholder="Ask about ticket prices or destinations...",
            lines=2,
            scale=4
        )

        send_button = gr.Button(
            "Send",
            variant="primary",
            scale=1
        )


# Hooking up events to callbacks
    message.submit(put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
        chat, inputs=chatbot, outputs=[chatbot, audio_output, image_output]
    )
    send_button.click(
        put_message_in_chatbot,
        inputs=[message, chatbot],
        outputs=[message, chatbot]
    ).then(
        chat,
        inputs=chatbot,
        outputs=[chatbot, audio_output, image_output]
    )

ui.launch(inbrowser=True, auth=("sunny", "12345"))