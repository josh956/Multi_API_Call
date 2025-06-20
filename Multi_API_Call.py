import os
import streamlit as st
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .element-container {
        word-break: break-word !important;
    }
    .stMarkdown, .stText, .stTextInput, .stSubheader, .stHeader, .stTitle {
        word-break: break-word !important;
        white-space: pre-wrap !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Options")
    tab = st.radio(
        "Select a tab:",
        ["Main App", "Pricing Table", "Model Comparison"]
    )

# --- PRICING TABLE TAB ---
if tab == "Pricing Table":
    st.header("LLM Pricing Table (per 1M tokens)")
    st.write("Note: DeepSeek provides discounts on pricing based on time of day")

    pricing_data = {
        "OpenAI (GPT-4.1)": {"Input": "$2.00", "Output": "$8.00"},
        "Google Gemini (Gemini 2.5 Flash)": {"Input": "$0.30", "Output": "$2.50"},
        "DeepSeek (DeepSeek Chat)": {"Input": "$0.27", "Output": "$1.10"},
        "X.AI (Grok-3)": {"Input": "$3.00", "Output": "$15.00"},
        "Anthropic (Claude Sonnet 4)": {"Input": "$4.00", "Output": "$15.00"},
    }
    st.table(
        {
            "Model": list(pricing_data.keys()),
            "Input Price (per 1M)": [v["Input"] for v in pricing_data.values()],
            "Output Price (per 1M)": [v["Output"] for v in pricing_data.values()],
        }
    )

# --- MODEL COMPARISON TAB ---
elif tab == "Model Comparison":
    st.header("LLM Model Comparison")
    st.write("Pros and cons for each model across 6 categories.")

    categories = [
        "Accuracy",
        "Speed",
        "Cost",
        "Context Length",
        "Tool Integration",
        "Multimodal Support"
    ]
    models = [
        "OpenAI (GPT-4.1)",
        "Google Gemini (Gemini 2.5 Flash)",
        "DeepSeek (DeepSeek Chat",
        "X.AI (Grok-3)",
        "Anthropic (Claude Sonnet 4)"
    ]
    pros_cons = {
        "OpenAI (GPT-4.1)": [
            "Very high", "Fast", "Moderate", "Large", "Excellent", "Good"
        ],
        "Google Gemini (Gemini 2.5 Flash)": [
            "High", "Very fast", "Low", "Large", "Good", "Excellent"
        ],
        "DeepSeek (DeepSeek Chat)": [
            "Good", "Fast", "Low", "Medium", "Fair", "Limited"
        ],
        "X.AI (Grok-3)": [
            "Good", "Moderate", "Low", "Medium", "Limited", "Limited"
        ],
        "Anthropic (Claude Sonnet 4)": [
            "High", "Fast", "Moderate", "Very large", "Good", "Good"
        ],
    }
    df = pd.DataFrame(pros_cons, index=categories)
    st.dataframe(df)

# --- MAIN APP TAB ---
else:
    # Get API keys from environment or Streamlit secrets
    OPENAI_API_KEY = os.getenv("General") if os.getenv("General") else st.secrets["General"]["key"]
    GOOGLE_API_KEY = os.getenv("GOOGLE_GENERAL") if os.getenv("GOOGLE_GENERAL") else st.secrets["GOOGLE_GENERAL"]["key"]
    GROK_API_KEY = os.getenv("GROK_GENERAL") if os.getenv("GROK_GENERAL") else st.secrets["GROK_GENERAL"]["key"]
    CLAUDE_API_KEY = os.getenv("ANTHROPIC_GENERAL") if os.getenv("ANTHROPIC_GENERAL") else st.secrets["ANTHROPIC_GENERAL"]["key"]
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_GENERAL") if os.getenv("DEEPSEEK_GENERAL") else st.secrets["DEEPSEEK_GENERAL"]["key"]

    MODEL_CONFIGS = {
        "OpenAI (GPT-4.1)": {
            "api_key": OPENAI_API_KEY,
            "base_url": None,
            "model": "gpt-4.1"
        },
        "Google Gemini (Gemini 2.5 Flash)": {
            "api_key": GOOGLE_API_KEY,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "model": "gemini-2.5-flash"
        },
        "DeepSeek (DeepSeek Chat)": {
            "api_key": DEEPSEEK_API_KEY,
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat"
        },
        "X.AI (Grok-3)": {
            "api_key": GROK_API_KEY,
            "base_url": "https://api.x.ai/v1",
            "model": "grok-3"
        },
        "Anthropic (Claude Sonnet 4)": {
            "api_key": CLAUDE_API_KEY,
            "base_url": "https://api.anthropic.com/v1/",
            "model": "claude-sonnet-4-20250514"
        }
    }

    st.header("LLM API Comparison")
    st.write("Choose one or more models/tools to generate and compare responses.")

    model_choices = st.multiselect(
        "Select the models/tools:",
        options=[
            "OpenAI (GPT-4.1)",
            "Google Gemini (Gemini 2.5 Flash)",
            "DeepSeek (DeepSeek Chat)",
            "X.AI (Grok-3)",
            "Anthropic (Claude Sonnet 4)"
        ],
        default=["OpenAI (GPT-4.1)"]
    )

    prompt = st.text_input("Enter your prompt:", placeholder="Type something here...")

    def get_model_response(model_choice, prompt):
        system_prompt = "You are a helpful assistant. Answer the users questions in 5 sentences or less"
        
        config = MODEL_CONFIGS.get(model_choice)
        if not config:
            return model_choice, "Unknown model selected."

        try:
            client_params = {"api_key": config["api_key"]}
            if config["base_url"]:
                client_params["base_url"] = config["base_url"]
            
            client = OpenAI(**client_params)
            
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            return model_choice, response.choices[0].message.content
        except Exception as e:
            return model_choice, f"An error occurred: {e}"

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

            if results:
                cols = st.columns(len(model_choices))
                result_dict = dict(results)
                for idx, model_choice in enumerate(model_choices):
                    with cols[idx]:
                        st.subheader(f"{model_choice} Response:")
                        st.markdown(f"<div style='font-family: sans-serif; white-space: pre-wrap'>{result_dict.get(model_choice, 'No response')}</div>", unsafe_allow_html=True)
