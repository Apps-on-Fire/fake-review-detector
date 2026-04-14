# Fake Review Detector - MCP Server

MCP server for fake review detection using a hybrid architecture:
- **Supervised classifier** (TF-IDF + LogisticRegression) decides the label
- **Pinecone** retrieves similar reviews for context
- **LLM** (GPT-4o-mini) explains the decision
- **LangGraph** orchestrates the deterministic workflow
- **MCP** (Model Context Protocol) exposes the system as tools
- **Dataset**: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset

> Core principle: the truth comes from the supervised model; the explanation comes from the LLM.

## Architecture

```
User request
    |
    v
[MCP Server] --> [LangGraph Workflow]
                      |
                      v
                 [Preprocess] --> [Classify (TF-IDF + LR)] --> [Retrieve (Pinecone)] --> [Explain (GPT-4o-mini)] --> [Format]
                                        |                            |                          |
                                   label + confidence          similar reviews            natural language
                                                                                          explanation
```

## Tools

- `analyze_review(text, category, rating)` - Full classification pipeline
- `get_similar_reviews(text, category, top_k)` - Similarity search + pattern analysis

## Prerequisites

- Python 3.12+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Pinecone API key](https://app.pinecone.io/)

## Setup

```bash
# Clone the repository
git clone https://github.com/Apps-on-Fire/fake-review-detector.git
cd fake-review-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your API keys:
#   OPENAI_API_KEY=sk-...
#   PINECODE_API_KEY=pcsk_...
```

## Offline Training

These steps only need to be run once (or when you want to retrain the model).

### 1. Prepare the dataset

Place your `reviews.csv` file in the project root. The CSV must have columns: `category`, `rating`, `text_original`, `label` (CG = fake, OR = real).

Split into train/test sets (80/20):

```bash
python split_dataset.py
```

This creates:
- `treinamento/reviews_treino.csv` (training set)
- `testes/reviews_teste.csv` (test set)

### 2. Train the classifier

```bash
python -m app.offline.train
```

This trains a TF-IDF + LogisticRegression pipeline and saves the model to `app/ml/artifacts/classifier.joblib`.

### 3. Index reviews in Pinecone

Make sure your Pinecone index `fake-reviews` exists (1536 dimensions, cosine metric).

```bash
python -m app.offline.build_index
```

This generates embeddings (OpenAI text-embedding-3-small) for all training reviews and indexes them in Pinecone.

### 4. Evaluate (optional)

```bash
python evaluate_errors.py
```

Saves misclassified reviews to `testes/misclassified_reviews.csv`.

## Running Locally

### Start the MCP server (stdio)

```bash
python -m app.server.mcp_server
```

### Using in VS Code

Add to your `.mcp.json` (project root or `~/.vscode/settings.json`):

```json
{
  "mcpServers": {
    "fake-review-detector": {
      "command": "/path/to/your/.venv/bin/python",
      "args": ["-m", "app.server.mcp_server"],
      "cwd": "/path/to/fake-review-detector"
    }
  }
}
```

After saving, reload VS Code (`Ctrl+Shift+P` > `Developer: Reload Window`). The tools will appear in the Copilot/Claude chat.

## Deploy to Hugging Face Spaces

### 1. Create a Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Docker** as the SDK (empty template)
3. Set visibility to **Public** (required for external access)

### 2. Push the code

```bash
# Add HF remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/fake-review-detector

# Push
git push hf main
```

### 3. Configure secrets

In your Space's **Settings > Variables and secrets**, add:

| Secret | Value |
|--------|-------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `PINECODE_API_KEY` | Your Pinecone API key |
| `MCP_AUTH_TOKEN` | A strong password to protect the tools |

The Space will rebuild automatically. Wait until the status shows **Running**.

### 4. Use the remote server in VS Code

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "fake-review-detector-remote": {
      "url": "https://YOUR_USERNAME-fake-review-detector.hf.space/sse"
    }
  }
}
```

Reload VS Code to connect.

### 5. Use in Claude.ai

1. Go to [claude.ai](https://claude.ai)
2. Open a chat and look for **Add custom connector** (or MCP settings)
3. Add the URL: `https://YOUR_USERNAME-fake-review-detector.hf.space/sse`
4. Leave OAuth fields empty (auth is handled by the tool's `auth_token` parameter)

When using the tools, pass your `MCP_AUTH_TOKEN` as the `auth_token` parameter:

> "Analyze this review with auth_token 'YOUR_TOKEN': the product was terrible quality..."

## Project Structure

```
├── app/
│   ├── server/mcp_server.py       # MCP server (stdio + SSE)
│   ├── graph/
│   │   ├── workflow.py            # LangGraph workflow
│   │   ├── state.py               # Shared state definition
│   │   └── nodes/                 # Workflow nodes
│   │       ├── preprocess_node.py
│   │       ├── classify_node.py
│   │       ├── retrieve_node.py
│   │       ├── explain_node.py
│   │       └── format_node.py
│   ├── ml/
│   │   ├── classifier.py          # Classifier wrapper
│   │   ├── preprocessing.py       # Text cleaning
│   │   └── artifacts/             # Trained model files
│   ├── llm/explainer.py           # GPT-4o-mini explainer
│   ├── retrieval/pinecone_client.py # Pinecone vector search
│   └── offline/
│       ├── train.py               # Model training script
│       └── build_index.py         # Pinecone indexing script
├── split_dataset.py               # Dataset splitting script
├── evaluate_errors.py             # Error analysis script
├── Dockerfile                     # Container for HF Spaces
├── requirements.txt
└── .env.example
```

## License

MIT
