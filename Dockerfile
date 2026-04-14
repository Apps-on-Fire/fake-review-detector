FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV MCP_TRANSPORT=sse

EXPOSE 7860

CMD ["python", "-m", "app.server.mcp_server"]
