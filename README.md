# FastAPI LLM App

A FastAPI application that provides a unified interface for LLM providers, managed with `uv` and containerized with Docker.

## Features

- FastAPI web framework
- LLM provider abstraction layer
- Docker and Docker Compose setup
- Health check endpoint
- WebSocket support for real-time updates
- Streaming support for LLM responses

## Prerequisites

- Docker and Docker Compose installed
- `uv` installed (optional, for local development)

## Quick Start

### Using Docker Compose (Recommended)

1. Create a `.env` file in the project root:

```bash
# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=app.log
LOG_DIR=logs
```

2. Build and start the service:

```bash
docker-compose up --build
```

3. The application will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Root endpoint (serves web UI)
- `GET /health` - Health check endpoint
- `GET /api/llm/providers` - List all available LLM providers
- `GET /api/llm/health` - Check LLM provider health
- `POST /api/llm/generate` - Generate text using LLM
- `POST /api/llm/stream` - Stream text generation
- `WS /api/llm/ws/status` - WebSocket for real-time status updates

### Example Usage

```bash
# Check health
curl "http://localhost:8000/health"

# List providers
curl "http://localhost:8000/api/llm/providers"

# Generate text
curl -X POST "http://localhost:8000/api/llm/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "provider": "groq", "temperature": 0.7}'
```

### Local Development (without Docker)

1. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:

```bash
uv pip install -e .
```

3. Create `.env` file:

```bash
cp .env.example .env
```

4. Run the application:

```bash
uvicorn fastapi_app.main:app --reload
```

## Project Structure

```
.
├── fastapi_app/        # Application package
│   ├── __init__.py    # Package initialization
│   ├── main.py        # FastAPI application
│   ├── config.py      # Configuration management
│   ├── logging_config.py  # Logging setup
│   ├── llm/           # LLM abstraction layer
│   │   ├── base.py    # Base LLM class
│   │   └── __init__.py
│   ├── routes/        # API routes
│   │   └── llm.py     # LLM endpoints
│   └── static/        # Web UI
│       └── index.html
├── pyproject.toml      # Python dependencies (uv)
├── Dockerfile          # Docker image configuration
├── docker-compose.yml  # Docker Compose configuration
├── .dockerignore       # Files to ignore in Docker build
├── .env.example        # Environment variables example
└── README.md           # This file
```

## Environment Variables

- `LOG_LEVEL` - Logging level (default: INFO)
- `LOG_FILE` - Log file name (default: app.log)
- `LOG_DIR` - Log directory (default: logs)

**Note:** The `.env` file is automatically loaded by docker-compose.yml. Make sure to create it before starting the services.

## API Documentation

Once the app is running, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
