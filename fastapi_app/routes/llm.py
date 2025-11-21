"""LLM API routes - FIXED VERSION"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Set
import time
import asyncio
import json

from ..llm import BaseLLM, LLMState, PROVIDERS
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/llm", tags=["LLM"])

llm_instances = PROVIDERS
stats_websockets: Set[WebSocket] = set()


async def broadcast_stats(stats: dict):
    """Send real-time generation stats to all connected frontend clients"""
    if not stats_websockets:
        return
    message = {"type": "generation_stats", **stats}
    disconnected = set()
    for ws in stats_websockets:
        try:
            if ws.client_state.name == "CONNECTED":
                await ws.send_json(message)
            else:
                disconnected.add(ws)
        except Exception:
            disconnected.add(ws)
    stats_websockets.difference_update(disconnected)


def get_llm_instance(provider: str) -> BaseLLM:
    if provider not in llm_instances:
        available = ", ".join(llm_instances.keys())
        raise HTTPException(status_code=400, detail=f"Provider '{provider}' not found. Available: {available}")
    return llm_instances[provider]


class GenerateRequest(BaseModel):
    prompt: str
    provider: str = "groq"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    stream: bool = False  # FIX: Added stream parameter


@router.get("/providers")
async def list_providers():
    """Return all available LLM providers with current state"""
    result = []
    for name, llm in llm_instances.items():
        state = llm.validate_config()
        provider_info = {
            "provider": name,
            "model": llm.model,
            "state": state.value,
        }
        if llm.error_message:
            provider_info["error_message"] = llm.error_message
        result.append(provider_info)
    return {"providers": result}


@router.get("/health")
async def llm_health(provider: Optional[str] = None, check: bool = True):
    """Get health status"""
    if provider:
        if provider not in llm_instances:
            raise HTTPException(status_code=404, detail="Provider not found")
        llm = llm_instances[provider]
        if check:
            await llm.check_health()
        return {
            "provider": provider,
            "state": llm.state.value,
            "error_message": llm.error_message,
        }

    if check:
        await perform_health_checks_for_all_providers()

    results = {}
    all_healthy = True
    any_configured = False

    for name, llm in llm_instances.items():
        state = llm.state
        results[name] = {
            "state": state.value,
            "error_message": llm.error_message,
        }
        if state == LLMState.HEALTHY:
            any_configured = True
        if state != LLMState.HEALTHY:
            all_healthy = False

    status = "healthy" if all_healthy and any_configured else "unhealthy" if any_configured else "not_configured"
    return {"status": status, "providers": results}


@router.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text - handles both streaming and non-streaming"""
    llm = get_llm_instance(request.provider)

    if not llm.is_configured():
        raise HTTPException(status_code=503, detail=f"Provider '{request.provider}' not configured (missing API key)")

    # FIX: Route to streaming if requested
    if request.stream:
        return await stream_response(request)

    start_time = time.time()
    try:
        response = await llm.generate(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        elapsed = time.time() - start_time

        # Update state on success
        if llm.state != LLMState.HEALTHY:
            llm._set_state(LLMState.HEALTHY)

        approx_tokens = len(response.content.split())
        await broadcast_stats({
            "provider": request.provider,
            "tokens": approx_tokens,
            "total_time": round(elapsed, 2),
            "tokens_per_second": round(approx_tokens / elapsed if elapsed > 0 else 0, 2),
        })

        return response

    except Exception as e:
        llm._set_state(LLMState.UNHEALTHY, str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(request: GenerateRequest):
    """Internal streaming handler"""
    llm = get_llm_instance(request.provider)

    async def event_generator():
        start = time.time()
        first_token = True
        full_text = ""
        token_count = 0

        try:
            async for chunk in llm.generate_stream(
                prompt=request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                if chunk:
                    full_text += chunk
                    token_count += 1

                    if first_token:
                        ttft = time.time() - start
                        await broadcast_stats({
                            "provider": request.provider,
                            "time_to_first_token": round(ttft, 3),
                        })
                        first_token = False

                    yield f"data: {json.dumps({'delta': chunk})}\n\n"

            total_time = time.time() - start
            tps = token_count / total_time if total_time > 0 else 0

            # Update state on success
            if llm.state != LLMState.HEALTHY:
                llm._set_state(LLMState.HEALTHY)

            await broadcast_stats({
                "provider": request.provider,
                "tokens": token_count,
                "total_time": round(total_time, 2),
                "tokens_per_second": round(tps, 2),
            })

            yield f"data: {json.dumps({'done': True, 'content': full_text})}\n\n"

        except Exception as e:
            llm._set_state(LLMState.UNHEALTHY, str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/stream")
async def stream(request: GenerateRequest):
    """Dedicated streaming endpoint"""
    llm = get_llm_instance(request.provider)
    if not llm.is_configured():
        raise HTTPException(status_code=503, detail=f"Provider '{request.provider}' not configured")
    request.stream = True
    return await stream_response(request)


@router.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    await websocket.accept()
    stats_websockets.add(websocket)
    try:
        await push_all_providers_status(websocket)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        stats_websockets.discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        stats_websockets.discard(websocket)


async def perform_health_checks_for_all_providers():
    async def check_one(name: str, llm: BaseLLM):
        if llm.is_configured() and llm.state != LLMState.HEALTHY:
            await llm.check_health()
    tasks = [check_one(name, llm) for name, llm in llm_instances.items()]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def push_all_providers_status(websocket: WebSocket):
    results = {}
    for name, llm in llm_instances.items():
        state = llm.validate_config()
        results[name] = {
            "state": state.value,
            "model": llm.model,
            "error_message": llm.error_message if state != LLMState.HEALTHY else None
        }
    overall = "healthy" if all(v["state"] == "healthy" for v in results.values()) else "partial"
    try:
        await websocket.send_json({"type": "status", "status": overall, "providers": results})
    except:
        pass


async def close_all_websockets():
    for ws in list(stats_websockets):
        try:
            await ws.close()
        except:
            pass
    stats_websockets.clear()