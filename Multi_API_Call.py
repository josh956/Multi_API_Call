import os
import streamlit as st
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

st.set_page_config(layout="wide")
# Get API keys from environment or Streamlit secrets
OPENAI_API_KEY = os.getenv("General") if os.getenv("General") else st.secrets["General"]["key"]
GOOGLE_API_KEY = os.getenv("GOOGLE_GENERAL") if os.getenv("GOOGLE_GENERAL") else st.secrets["GOOGLE_GENERAL"]["key"]
GROK_API_KEY = os.getenv("GROK_GENERAL") if os.getenv("GROK_GENERAL") else st.secrets["GROK_GENERAL"]["key"]
CLAUDE_API_KEY = os.getenv("ANTHROPIC_GENERAL") if os.getenv("ANTHROPIC_GENERAL") else st.secrets["ANTHROPIC_GENERAL"]["key"]
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_GENERAL") if os.getenv("DEEPSEEK_GENERAL") else st.secrets["DEEPSEEK_GENERAL"]["key"]

def get_model_response(model_choice, prompt):
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
            return model_choice, response.choices[0].message.content

        # Google Gemini (Gemini 2.5 Flash)
        elif model_choice == "Google Gemini (Gemini 2.5 Flash)":
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
            return model_choice, response.choices[0].message.content

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
            return model_choice, response.choices[0].message.content

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
            return model_choice, response.choices[0].message.content

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
            return model_choice, response.choices[0].message.content

    except Exception as e:
        return model_choice, f"An error occurred with {model_choice}: {e}"

# Streamlit UI
st.title("LLM API Comparison")
st.write("Choose a model and tool to generate a response.")

# Dropdown for model selection
model_choices = st.multiselect(
    "Select the models/tools:",
    options=[
        "OpenAI (GPT-4.1)",
        "Google Gemini (Gemini 2.5 Flash)",
        "DeepSeek (DeepSeek Chat)",
        "X.AI (Grok-3)",
        "Anthropic (Claude Sonnet 3)"
    ],
    default=["OpenAI (GPT-4.1)"]  # Optional: pre-select one
)

# Input for user prompt
prompt = st.text_input("Enter your prompt:", placeholder="Type something here...")

# Button to generate response
if st.button("Generate Response"):
    if not prompt:
        st.error("Please enter a prompt.")
    elif not model_choices:
        st.error("Please select at least one model.")
    else:
        results = []
        with ThreadPoolExecutor(max_workers=len(model_choices)) as executor:
            future_to_model = {executor.submit(get_model_response, model_choice, prompt): model_choice for model_choice in model_choices}
            for future in as_completed(future_to_model):
                model_choice, response_text = future.result()
                results.append((model_choice, response_text))

        # Display results in the order of model_choices
        if results:
            cols = st.columns(len(model_choices))
            # Map model name to its response for easy lookup
            result_dict = dict(results)
            max_cols_per_row = 3  # or 2, or whatever you prefer
            num_models = len(model_choices)
            num_rows = math.ceil(num_models / max_cols_per_row)

            for row in range(num_rows):
                cols = st.columns(min(max_cols_per_row, num_models - row * max_cols_per_row))
                for col_idx, model_idx in enumerate(range(row * max_cols_per_row, min((row + 1) * max_cols_per_row, num_models))):
                    model_choice = model_choices[model_idx]
                    with cols[col_idx]:
                        st.subheader(f"{model_choice} Response:")
                        st.write(result_dict.get(model_choice, "No response"))
