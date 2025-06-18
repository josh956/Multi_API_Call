import os
import streamlit as st
from openai import OpenAI

# Get API keys from environment or Streamlit secrets

OPENAI_API_KEY = os.getenv("General") if os.getenv("General") else st.secrets["General"]["key"]
GOOGLE_API_KEY = os.getenv("GOOGLE_GENERAL") if os.getenv("GOOGLE_GENERAL") else st.secrets["GOOGLE_GENERAL"]["key"]
GROK_API_KEY = os.getenv("GROK_GENERAL") if os.getenv("GROK_GENERAL") else st.secrets["GROK_GENERAL"]["key"]
CLAUDE_API_KEY = os.getenv("ANTHROPIC_GENERAL") if os.getenv("ANTHROPIC_GENERAL") else st.secrets["ANTHROPIC_GENERAL"]["key"]
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_GENERAL") if os.getenv("DEEPSEEK_GENERAL") else st.secrets["DEEPSEEK_GENERAL"]["key"]

# Streamlit UI
st.title("LLM API Comparison")
st.write("Choose a model and tool to generate a response.")

# Dropdown for model selection
model_choice = st.selectbox(
    "Select the model/tool:",
    options=[
        "OpenAI (GPT-4.1)",
        "Google Gemini (Gemini 2.5 Flash)",
        "DeepSeek (DeepSeek Chat)",
        "X.AI (Grok-3)",
        "Anthropic (Claude Sonnet 3)"
    ]
)

# Input for user prompt
prompt = st.text_input("Enter your prompt:", placeholder="Type something here...")

# Button to generate response
if st.button("Generate Response"):
    if not prompt:
        st.error("Please enter a prompt.")
    else:
        try:
            # OpenAI (GPT-4.1)
            if model_choice == "OpenAI (GPT-4.1)":
                client = OpenAI(api_key=OPENAI_API_KEY)
                model = "gpt-4.1"
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer the users questions"
                        "in 5 sentences or less"},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )
                st.subheader("OpenAI Response:")
                st.write(response.choices[0].message.content)

            # Google Gemini (Gemini 2.0 Flash)
            elif model_choice == "Google Gemini (Gemini 2.0 Flash)":
                client = OpenAI(
                    api_key=GOOGLE_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                model = "gemini-2.5-flash"
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer the users questions"
                        "in 5 sentences or less"},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )
                st.subheader("Google Gemini Response:")
                st.write(response.choices[0].message.content)

            # DeepSeek (DeepSeek Chat)
            elif model_choice == "DeepSeek (DeepSeek Chat)":
                client = OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com"
                )
                model = "deepseek-chat"
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer the users questions"
                        "in 5 sentences or less"},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )
                st.subheader("DeepSeek Response:")
                st.write(response.choices[0].message.content)

            # X.AI (Grok-3)
            elif model_choice == "X.AI (Grok-3)":
                client = OpenAI(
                    api_key=GROK_API_KEY,
                    base_url="https://api.x.ai/v1"
                )
                model = "grok-3"
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant.Answer the users questions"
                        "in 5 sentences or less"},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )
                st.subheader("X.AI Response:")
                st.write(response.choices[0].message.content)

            # Anthropic (Claude Sonnet 3)
            elif model_choice == "Anthropic (Claude Sonnet 3)":
                client = OpenAI(
                    api_key=CLAUDE_API_KEY,
                    base_url="https://api.anthropic.com/v1/"
                )
                model = "claude-3-7-sonnet-20250219"
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant.Answer the users questions"
                        "in 5 sentences or less"},
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )
                st.subheader("Anthropic Claude Response:")
                st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"An error occurred: {e}")
