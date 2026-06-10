# How to Deploy the Fake Review Detector MCP

This guide explains how to deploy the two infrastructure pieces this MCP server depends on:

1. **Pinecone** — the vector database that stores the indexed reviews and powers similarity search.
2. **AWS** — where the MCP server itself runs as a container (SSE transport over HTTP).

> The MCP server is **not** hosted on Pinecone. Pinecone is the vector store; AWS is the host. Part 1 sets up Pinecone, Part 2 deploys the server to AWS and points it at that Pinecone index.

---

## Architecture recap

```
MCP client (Claude.ai / VS Code)
        |  HTTPS (SSE)
        v
[ MCP Server container on AWS ]
        |
        |-- classifier.joblib (TF-IDF + LogisticRegression, bundled in image)
        |-- OpenAI API ............ embeddings + GPT-4o-mini explanations
        |-- Pinecone .............. similar-review retrieval
```

The server listens on **port 7860** and, when `MCP_TRANSPORT=sse`, exposes the MCP
endpoint at `/sse`.

### Required environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | local/offline only | Embeddings (`text-embedding-3-small`) + explanations (`gpt-4o-mini`). **The remote server does not use this by default** — each caller brings their own key via the `openai_api_key` tool argument (see [Bring-your-own OpenAI key](#bring-your-own-openai-key)). Still needed locally to build the Pinecone index (Part 1.3). Only consumed server-side if `ALLOW_SERVER_OPENAI_KEY` is enabled |
| `PINECODE_API_KEY` | yes | Pinecone access. **Note the spelling** — the code reads `PINECODE_API_KEY` (see [pinecone_client.py:11](app/retrieval/pinecone_client.py#L11)), not `PINECONE_API_KEY` |
| `MCP_TRANSPORT` | yes (remote) | Set to `sse` for HTTP deployment. Defaults to `stdio` for local use |
| `MCP_AUTH_TOKEN` | recommended | Shared secret. When set, callers must pass it as the `auth_token` tool argument. If empty, the tools are open |
| `ALLOW_SERVER_OPENAI_KEY` | no | When `1`/`true`, the server may fall back to its own `OPENAI_API_KEY` if a caller didn't pass one. **Leave unset in production** so callers never spend the host's OpenAI credits |

### Bring-your-own OpenAI key

By default the **remote server never spends the host's OpenAI credits**. Each caller
supplies their own key as the `openai_api_key` tool argument on the **first call of a
conversation**; the server keeps it in memory **for that MCP session only** and reuses it
on later calls (no need to resend). Without a valid key the tools return a
`missing_openai_key` error instead of falling back to the host's key. Pinecone stays the
deployer's because it holds the indexed dataset built in Part 1.

---

## Part 1 — Deploy / set up Pinecone

Pinecone holds the embeddings of the **training** reviews so the server can retrieve
similar reviews at query time. You set this up **once** (and again only when you retrain
or re-index).

### 1.1 Create a Pinecone account and API key

1. Sign up / log in at [app.pinecone.io](https://app.pinecone.io/).
2. Open **API Keys** and create a key (it looks like `pcsk_...`).
3. Save it — you will use it both locally (to build the index) and on AWS (to query it).

### 1.2 Index configuration (must match the code)

The retrieval code in [pinecone_client.py](app/retrieval/pinecone_client.py) and the
indexer in [build_index.py](app/offline/build_index.py) expect exactly:

| Setting | Value |
|---|---|
| Index name | `fake-reviews` |
| Dimension | `1536` (matches `text-embedding-3-small`) |
| Metric | `cosine` |
| Type | Serverless |
| Cloud / region | `aws` / `us-east-1` |

You can create the index two ways:

**Option A — automatically (recommended).** The build script creates the index for you
if it does not exist (see [build_index.py:37-56](app/offline/build_index.py#L37-L56)).
Skip to 1.3.

**Option B — manually in the console.** In the Pinecone dashboard click **Create index**,
name it `fake-reviews`, set dimension `1536`, metric `cosine`, choose **Serverless**,
cloud **AWS**, region **us-east-1**, then create.

### 1.3 Build (populate) the index

This step runs locally (or from any machine with the repo and Python). It reads the
training CSV, generates OpenAI embeddings, and upserts them into Pinecone.

```bash
# From the repository root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Provide both keys (used by the build script)
export OPENAI_API_KEY="sk-..."
export PINECODE_API_KEY="pcsk-..."

# Make sure the training set exists (run once if needed):
#   python split_dataset.py   -> creates treinamento/reviews_treino.csv

# Build the index: creates 'fake-reviews' if missing, then upserts all vectors
python -m app.offline.build_index
```

On success you will see progress lines and a final
`Indexação completa. Total de vetores: <N>`.

Each vector is stored with metadata used for filtering at query time:
`category`, `rating`, `label` (`CG` = fake, `OR` = real), `split` (`"train"`),
and a truncated `text`. The query path filters on `split = "train"` and optionally
`category` (see [pinecone_client.py:39-48](app/retrieval/pinecone_client.py#L39-L48)).

### 1.4 Verify

In the Pinecone console, open the `fake-reviews` index and confirm the vector count is
non-zero. The Pinecone side is now "deployed" — keep the `fake-reviews` index running and
note the `PINECODE_API_KEY` for the AWS deployment.

> Cost note: serverless Pinecone bills on storage + reads/writes. The free tier is enough
> for this dataset. Deleting the index removes the data; re-run 1.3 to rebuild.

---

## Part 2 — Deploy the MCP server on AWS

The repo already ships a [Dockerfile](Dockerfile) that builds an SSE server on port 7860
and bundles the trained classifier (`app/ml/artifacts/classifier.joblib`). We push that
image to **Amazon ECR** and run it. Three options are described, from simplest to most
control:

- **2A — AWS App Runner** (simplest; fully managed, public HTTPS URL out of the box)
- **2B — Amazon ECS on Fargate** (more control, behind a load balancer)
- **2C — Plain EC2** (quickest throwaway / dev box)

All three start from the same image in ECR (Part 2.1).

### Prerequisites

- An AWS account and the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed and configured (`aws configure`).
- Docker installed locally.
- The `OPENAI_API_KEY` and `PINECODE_API_KEY` values, and a chosen `MCP_AUTH_TOKEN`.
- Pinecone index `fake-reviews` already populated (Part 1).

Set some shell variables used throughout:

```bash
export AWS_REGION="us-east-1"
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export ECR_REPO="fake-review-detector"
export IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"
```

### 2.1 Build and push the image to Amazon ECR

```bash
# 1. Create the ECR repository (once)
aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION"

# 2. Authenticate Docker to ECR
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# 3. Build (use --platform for Apple Silicon / ARM laptops so it runs on x86 AWS)
docker build --platform linux/amd64 -t "$ECR_REPO" .

# 4. Tag and push
docker tag "${ECR_REPO}:latest" "$IMAGE_URI"
docker push "$IMAGE_URI"
```

> Storing secrets: for production, put `OPENAI_API_KEY`, `PINECODE_API_KEY`, and
> `MCP_AUTH_TOKEN` in **AWS Secrets Manager** or **SSM Parameter Store** and reference
> them from the service, rather than passing them as plain environment variables.

---

### 2A — Deploy with AWS App Runner (simplest)

App Runner pulls the image from ECR, runs it, gives you a public HTTPS URL, and handles
TLS and scaling. This is the closest equivalent to the Hugging Face Spaces deployment.

1. Open the **AWS App Runner** console → **Create service**.
2. **Source**: Container registry → Amazon ECR → browse to `fake-review-detector:latest`.
3. **Deployment**: Manual (or Automatic to redeploy on every image push).
4. **Service settings**:
   - **Port**: `7860`
   - **Environment variables**:
     - `MCP_TRANSPORT = sse`
     - `OPENAI_API_KEY = sk-...`
     - `PINECODE_API_KEY = pcsk-...`
     - `MCP_AUTH_TOKEN = <your-strong-token>`
   - (Or reference Secrets Manager values instead of plaintext.)
5. **Health check**: TCP on port `7860` is the most reliable for an SSE server.
6. Create the service. After a few minutes App Runner shows a **Default domain** like
   `https://xxxx.us-east-1.awsapprunner.com`.

Your MCP endpoint is that domain plus `/sse`:

```
https://xxxx.us-east-1.awsapprunner.com/sse
```

CLI equivalent (optional):

```bash
aws apprunner create-service \
  --service-name fake-review-detector \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "'"$IMAGE_URI"'",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "7860",
        "RuntimeEnvironmentVariables": {
          "MCP_TRANSPORT": "sse",
          "OPENAI_API_KEY": "sk-...",
          "PINECODE_API_KEY": "pcsk-...",
          "MCP_AUTH_TOKEN": "your-strong-token"
        }
      }
    },
    "AutoDeploymentsEnabled": false
  }' \
  --region "$AWS_REGION"
```

---

### 2B — Deploy with Amazon ECS on Fargate (more control)

Use this when you want VPC networking, autoscaling policies, and an Application Load
Balancer (ALB).

1. **Cluster**: ECS console → Create cluster → **Networking only (Fargate)**.

2. **Task definition** (Fargate, Linux/X86_64, e.g. 0.25 vCPU / 0.5 GB to start):
   - Container image: `$IMAGE_URI`
   - Port mapping: container port `7860` (TCP)
   - Environment variables: `MCP_TRANSPORT=sse`, `OPENAI_API_KEY`, `PINECODE_API_KEY`,
     `MCP_AUTH_TOKEN` (prefer `secrets` → Secrets Manager for the keys).
   - Log configuration: `awslogs` driver to CloudWatch.

3. **Service**:
   - Launch type Fargate, desired count `1`.
   - Attach an **Application Load Balancer** with a target group on port `7860`.
   - **Important for SSE**: increase the target group / ALB idle timeout (e.g. 300s+) so
     long-lived SSE streams are not cut. Set health check to a TCP/HTTP check on `7860`.
   - Security group: allow inbound `443` (ALB) from clients; the ALB forwards to the task.

4. Put **HTTPS** on the ALB with an ACM certificate (MCP clients require `https://`).
   Once the service is healthy, your endpoint is:

```
https://<your-alb-domain-or-custom-domain>/sse
```

---

### 2C — Quick deploy on a single EC2 instance (dev / throwaway)

Fastest path for testing; not recommended for production (no managed TLS or scaling).

1. Launch an EC2 instance (e.g. Amazon Linux 2023, `t3.small`). In its **security group**
   open inbound TCP `7860` (and `22` for SSH) to your IP.
2. SSH in and install Docker:

   ```bash
   sudo dnf install -y docker        # Amazon Linux 2023
   sudo systemctl enable --now docker
   ```
3. Authenticate to ECR and run the container:

   ```bash
   aws ecr get-login-password --region us-east-1 \
     | sudo docker login --username AWS --password-stdin \
       "<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com"

   sudo docker run -d --name fake-review-detector \
     -p 7860:7860 \
     -e MCP_TRANSPORT=sse \
     -e OPENAI_API_KEY="sk-..." \
     -e PINECODE_API_KEY="pcsk-..." \
     -e MCP_AUTH_TOKEN="your-strong-token" \
     "<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/fake-review-detector:latest"
   ```

   The instance needs an IAM role allowing ECR pull (`AmazonEC2ContainerRegistryReadOnly`),
   or run `docker login` with credentials.
4. Endpoint: `http://<EC2_PUBLIC_IP>:7860/sse`.

> For real use, front the instance with Nginx + a TLS certificate (or an ALB) so the
> endpoint is `https://`, which MCP clients require.

---

## Part 2-HF — Deploy on Hugging Face Spaces (alternative to AWS)

Hugging Face Spaces is the simplest fully-managed alternative to AWS: it builds the same
[Dockerfile](Dockerfile), runs it, and gives you a public HTTPS URL with TLS handled for
you — no ECR, load balancer, or certificate setup. Use this instead of Part 2 if you want
the quickest hosted deployment. (Pinecone setup in **Part 1** is still required.)

A Space uses the Docker SDK and exposes the container on port `7860`, which matches this
project's [Dockerfile](Dockerfile) exactly. The Space "card" (the YAML header that
configures it) is already provided in [README_HF.md](README_HF.md):

```yaml
---
title: Fake Review Detector MCP
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
---
```

### 2-HF.1 Create a Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Docker** as the SDK (empty template).
3. Set visibility to **Public** (required so external MCP clients can reach `/sse`).

### 2-HF.2 Push the code

The Space is a git repository. Add it as a remote and push. Make sure a `README.md` with
the Space-card YAML above is at the root of what you push — you can copy
[README_HF.md](README_HF.md) to `README.md` for the Space, or push the contents of the
[hf-deploy/](hf-deploy/) folder which already contains a Space card.

```bash
# Add the Hugging Face remote (replace YOUR_USERNAME)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/fake-review-detector

# Push
git push hf main
```

Spaces builds the Docker image automatically on every push. The image bundles the trained
classifier (`app/ml/artifacts/classifier.joblib`) and starts the server with
`MCP_TRANSPORT=sse` (set in the [Dockerfile](Dockerfile)), so no transport config is needed.

### 2-HF.3 Configure secrets

In the Space's **Settings → Variables and secrets**, add:

| Secret | Value |
|---|---|
| `PINECODE_API_KEY` | Your Pinecone API key (note the `D` spelling — see [pinecone_client.py:11](app/retrieval/pinecone_client.py#L11)) |
| `MCP_AUTH_TOKEN` | Optional — a strong shared secret to gate access. Leave empty for an open, bring-your-own-key Space |

> **Do not set `OPENAI_API_KEY`** on the Space. This server is **bring-your-own-key**: each
> caller supplies their own OpenAI key at call time (next section), so they spend their own
> credits, not yours. Setting it has no effect unless you also set `ALLOW_SERVER_OPENAI_KEY=1`,
> which would defeat the purpose. Pinecone stays yours because it holds the indexed dataset
> from Part 1.

Saving secrets triggers a rebuild. Wait until the Space status shows **Running**.

### 2-HF.3a How callers provide their OpenAI key

On the **first tool call** of a conversation, the caller passes their key as the
`openai_api_key` argument; the server keeps it in memory **for that MCP session only** and
reuses it on later calls, so it isn't resent each time. Without a valid key the tools
return a `missing_openai_key` error (they never fall back to the host's key). For example:

> "Analyze this review with openai_api_key 'sk-...': the product broke after one day,
> category Home_and_Kitchen_5, rating 1"

Subsequent calls in the same conversation can omit `openai_api_key`.

### 2-HF.4 Your endpoint

The SSE endpoint is the Space's `.hf.space` domain plus `/sse`:

```
https://YOUR_USERNAME-fake-review-detector.hf.space/sse
```

Use this URL in **Part 3** to connect clients. (This matches the
`fake-review-detector-remote` entry already in [.mcp.json](.mcp.json).)

> The container also serves a human-readable landing page at `/` and a health probe at
> `/health` (both return `200`), so the Space root no longer logs a `404` — only the MCP
> protocol lives at `/sse`.

---

## Part 3 — Connect a client to the deployed server

Once AWS gives you a public HTTPS URL, point your MCP client at `<URL>/sse`.

**VS Code** — add to `.mcp.json`:

```json
{
  "mcpServers": {
    "fake-review-detector-remote": {
      "url": "https://YOUR-AWS-DOMAIN/sse"
    }
  }
}
```

**Claude.ai** — add a custom connector with the URL `https://YOUR-AWS-DOMAIN/sse`.
Leave OAuth fields empty; authentication is handled by the tool's `auth_token` argument.

When `MCP_AUTH_TOKEN` is set, pass it when calling the tools, e.g.:

> "Analyze this review with auth_token 'YOUR_TOKEN': the product was terrible quality..."

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 'PINECODE_API_KEY'` in logs | Wrong variable name | The code reads `PINECODE_API_KEY` (with the `D`), not `PINECONE_API_KEY` |
| Empty / no similar reviews returned | Index not built or wrong index | Re-run `python -m app.offline.build_index` (Part 1.3); confirm vector count in Pinecone |
| Dimension mismatch error from Pinecone | Index not 1536-dim cosine | Recreate `fake-reviews` with dimension `1536`, metric `cosine` |
| Client connects then drops the stream | Load balancer idle timeout too low for SSE | Raise ALB/target-group idle timeout (Part 2B) |
| 401 / "Unauthorized. Invalid auth_token." | `MCP_AUTH_TOKEN` mismatch | Pass the exact `MCP_AUTH_TOKEN` value as the `auth_token` argument |
| `missing_openai_key` error from a tool | No `openai_api_key` passed and none cached for the session | Pass your OpenAI key as the `openai_api_key` argument on the first call of the conversation; later calls reuse it |
| Space root logs `404 Not Found` on `GET /` | Image predates the `/` landing route | Redeploy a build that includes [mcp_server.py](app/server/mcp_server.py)'s `/` and `/health` routes |
| Image runs locally but fails on AWS | Built for ARM on an Apple Silicon Mac | Rebuild with `docker build --platform linux/amd64 ...` |
| OpenAI auth/quota errors | Invalid `openai_api_key` or no credit on that account | Verify the key the caller passed and its account billing (it's the caller's key, not the host's) |
