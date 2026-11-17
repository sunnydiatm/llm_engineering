import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import base64
from io import BytesIO
from PIL import Image

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


def talker(message):
    print("Calling LLM with Voice...")
    openai = OpenAI(api_key=chatgpt_api_key, base_url="https://api.openai.com/v1")
    response = openai.audio.speech.create(model="gpt-4o-mini-tts",
      voice="onyx",    # Also, try replacing onyx with alloy or coral
      input=message)

    # Save generated audio
    response.stream_to_file("output.mp3")

    print("Audio saved successfully: output.mp3")


talker("Hi there, how are you today?")

# run "afplay output.mp3" on terminal to hear the speech