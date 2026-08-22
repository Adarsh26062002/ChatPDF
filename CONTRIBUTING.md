# Contributing

Thanks for your interest in ChatPDF! All contributions are welcome.

## Setup

```bash
git clone https://github.com/kairav7220/ChatPDF.git
cd ChatPDF
pip install -r requirements.txt
```

Set your API keys in `.env`:

```env
MISTRAL_API_KEY="your-mistral-key"
MISTRAL_API_KEY_2="your-second-mistral-key"
GROQ_API_KEY="your-groq-key"
GOOGLE_API_KEY="your-google-gemini-key"
HUGGINGFACE_API_TOKEN="your-hf-token"
TAVILY_API_KEY="your-tavily-key"
```

Run the app:

```bash
streamlit run frontend.py
```

## Development

- Fork the repo and create a feature branch.
- Keep the split clean: UI in `frontend.py`, logic in `backend.py`.
- The CRAG pipeline lives in `backend.py` as a compiled LangGraph `StateGraph`. If you add or modify nodes, make sure the graph still compiles and routes correctly.
- `.streamlit/secrets.toml` and `.env` are both gitignored — never commit API keys.
- Verify your changes run with `streamlit run frontend.py`.

## PR Guidelines

- One feature or fix per PR.
- Describe what the change does and how you tested it.
- Keep the diff focused — avoid unrelated renames or formatting churn.
- Update the README if your change affects setup, structure, or features.
