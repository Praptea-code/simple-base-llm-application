"""LLM API routes"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal, Set
import time
import asyncio
from ..llm import GroqLLM, BaseLLM, LLMState
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/llm", tags=["LLM"])

# Initialize LLM instances
llm_instances: dict[str, BaseLLM] = {}

# WebSocket connections for stats updates
stats_websockets: Set[WebSocket] = set()


async def broadcast_stats(stats: dict):
    """Broadcast stats to all connected WebSocket clients"""
    if not stats_websockets:
        return

    message = {"type": "generation_stats", **stats}

    disconnected = set()
    for ws in stats_websockets:
        try:
            # Check if WebSocket is still open before sending
            if hasattr(ws, "client_state") and ws.client_state.name == "CONNECTED":
                await ws.send_json(message)
            else:
                # WebSocket is not connected, mark for removal
                disconnected.add(ws)
        except (RuntimeError, ConnectionError, WebSocketDisconnect, OSError) as e:
            # WebSocket is closed or error occurred
            error_msg = str(e).lower()
            # Ignore "bad file descriptor" errors during shutdown
            if "bad file descriptor" not in error_msg:
                logger.debug(f"Failed to send stats to WebSocket: {e}")
            disconnected.add(ws)
        except Exception as e:
            # Other errors - check if it's a connection-related error
            error_msg = str(e).lower()
            if "bad file descriptor" in error_msg:
                # Ignore during shutdown
                disconnected.add(ws)
            elif any(
                keyword in error_msg
                for keyword in ["close", "closed", "disconnect", "send"]
            ):
                logger.debug(f"Connection error sending stats to WebSocket: {e}")
                disconnected.add(ws)
            else:
                # Unexpected error - log but don't remove
                logger.debug(f"Unexpected error sending stats to WebSocket: {e}")

    # Remove disconnected WebSockets
    stats_websockets.difference_update(disconnected)


# Initialize Groq LLM
try:
    groq_llm = GroqLLM()
    llm_instances["groq"] = groq_llm
    logger.info(f"Groq LLM initialized - State: {groq_llm.state.value}")
except Exception as e:
    groq_llm = None
    logger.warning(f"Groq LLM initialization failed: {e}")


def get_llm_instance(provider: str) -> BaseLLM:
    """
    Get LLM instance for the specified provider

    Args:
        provider: Provider name ("groq")

    Returns:
        LLM instance

    Raises:
        HTTPException: If provider is not available
    """
    if provider not in llm_instances:
        available = ", ".join(llm_instances.keys())
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider '{provider}' is not configured. Available providers: {available}",
        )
    return llm_instances[provider]


class GenerateRequest(BaseModel):
    """Request model for LLM generation"""

    prompt: str = Field(..., description="The prompt to send to the LLM")
    provider: Literal["groq"] = Field(
        default="groq", description="LLM provider to use"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, gt=0, description="Maximum tokens to generate"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")


@router.get("/providers")
async def list_providers():
    """
    List all available LLM providers and their models

    Returns information about configured providers, their models, and current state.
    """
    providers_info = []

    for provider_name, llm_instance in llm_instances.items():
        state = llm_instance.validate_config()
        providers_info.append(
            {
                "provider": provider_name,
                "model": llm_instance.model,
                "state": state.value,
                "configured": llm_instance.is_configured(),
                "error_message": llm_instance.error_message
                if state != LLMState.HEALTHY
                else None,
            }
        )

    return {
        "providers": providers_info,
        "count": len(providers_info),
    }


@router.get("/health")
async def llm_health(provider: Optional[str] = None, check: bool = False):
    """
    Check LLM service health

    Args:
        provider: Optional provider name to check specific provider
        check: If True, perform actual health check (test API call), otherwise use cached state
    """
    if provider:
        # Check specific provider
        if provider not in llm_instances:
            return {
                "state": LLMState.NOT_CONFIGURED.value,
                "provider": provider,
                "model": None,
                "error_message": f"Provider '{provider}' is not configured",
            }
        llm = llm_instances[provider]

        # Perform health check if requested
        if check:
            state = await llm.check_health()
        else:
            state = llm.validate_config()

        return {
            "state": state.value,
            "provider": provider,
            "model": llm.model,
            "error_message": llm.error_message,
        }
    else:
        # Return health for all providers
        health_status = {}
        for prov, llm_instance in llm_instances.items():
            # Perform health check if requested
            if check:
                state = await llm_instance.check_health()
            else:
                state = llm_instance.validate_config()

            health_status[prov] = {
                "state": state.value,
                "model": llm_instance.model,
                "error_message": llm_instance.error_message,
            }

        # Determine overall status
        states = [h["state"] for h in health_status.values()]
        if all(s == LLMState.HEALTHY.value for s in states):
            overall_status = "healthy"
        elif any(s == LLMState.HEALTHY.value for s in states):
            overall_status = "partial"
        elif any(s == LLMState.UNHEALTHY.value for s in states):
            overall_status = "unhealthy"
        else:
            overall_status = "not_configured"

        return {
            "status": overall_status,
            "providers": health_status,
        }


@router.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate text using the LLM"""
    # Get the LLM instance for the specified provider
    llm = get_llm_instance(request.provider)

    logger.info(
        f"Generating text - provider: {request.provider}, stream: {request.stream}, "
        f"temperature: {request.temperature}, prompt: {request.prompt[:50]}..."
    )

    # Check provider state before generating
    if llm.state == LLMState.NOT_CONFIGURED:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider '{request.provider}' is not configured. Please set the API key.",
        )

    try:
        if request.stream:
            # Return streaming response with stats tracking
            import uuid

            request_id = str(uuid.uuid4())
            start_time = time.time()
            first_token_time = None
            input_tokens = 0
            output_tokens = 0
            last_stats_update = start_time
            stats_update_interval = 0.1  # Update stats every 100ms

            async def generate_stream():
                nonlocal first_token_time, output_tokens, last_stats_update

                # Estimate input tokens (rough approximation: ~4 chars per token)
                # For more accurate counting, we'd need to use tiktoken or similar
                input_tokens = len(request.prompt) // 4

                # Send initial stats
                await broadcast_stats(
                    {
                        "request_id": request_id,
                        "provider": request.provider,
                        "input_tokens": input_tokens,
                        "output_tokens": 0,
                        "tokens_per_second": 0.0,
                        "time_to_first_token": None,
                        "elapsed_time": 0.0,
                        "status": "generating",
                    }
                )

                chunk_count = 0
                total_chars = 0
                async for chunk in llm.generate_stream(
                    prompt=request.prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ):
                    current_time = time.time()

                    # Track first token time
                    if first_token_time is None:
                        first_token_time = current_time
                        time_to_first = first_token_time - start_time
                        await broadcast_stats(
                            {
                                "request_id": request_id,
                                "provider": request.provider,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "tokens_per_second": 0.0,
                                "time_to_first_token": round(time_to_first, 3),
                                "elapsed_time": round(time_to_first, 3),
                                "status": "streaming",
                            }
                        )

                    # Estimate output tokens (rough approximation: ~4 chars per token)
                    # Accumulate characters for better estimation
                    total_chars += len(chunk)
                    output_tokens = total_chars // 4
                    chunk_count += 1

                    # Update stats periodically (every 100ms)
                    if current_time - last_stats_update >= stats_update_interval:
                        elapsed = current_time - start_time
                        if elapsed > 0 and first_token_time:
                            # Calculate tokens per second from first token
                            streaming_time = current_time - first_token_time
                            if streaming_time > 0:
                                tokens_per_second = output_tokens / streaming_time
                            else:
                                tokens_per_second = 0.0
                        else:
                            tokens_per_second = 0.0

                        await broadcast_stats(
                            {
                                "request_id": request_id,
                                "provider": request.provider,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "tokens_per_second": round(tokens_per_second, 2),
                                "time_to_first_token": round(
                                    first_token_time - start_time, 3
                                )
                                if first_token_time
                                else None,
                                "elapsed_time": round(elapsed, 3),
                                "status": "streaming",
                            }
                        )
                        last_stats_update = current_time

                    yield f"data: {chunk}\n\n"

                # Send final stats
                final_time = time.time()
                total_elapsed = final_time - start_time
                streaming_time = (
                    final_time - first_token_time if first_token_time else total_elapsed
                )
                final_tokens_per_second = (
                    output_tokens / streaming_time if streaming_time > 0 else 0.0
                )

                await broadcast_stats(
                    {
                        "request_id": request_id,
                        "provider": request.provider,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "tokens_per_second": round(final_tokens_per_second, 2),
                        "time_to_first_token": round(first_token_time - start_time, 3)
                        if first_token_time
                        else None,
                        "elapsed_time": round(total_elapsed, 3),
                        "status": "completed",
                    }
                )

                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Return complete response with stats tracking
            import uuid

            request_id = str(uuid.uuid4())
            start_time = time.time()

            # Estimate input tokens
            input_tokens = len(request.prompt) // 4

            # Send initial stats
            await broadcast_stats(
                {
                    "request_id": request_id,
                    "provider": request.provider,
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "tokens_per_second": 0.0,
                    "time_to_first_token": None,
                    "elapsed_time": 0.0,
                    "status": "generating",
                }
            )

            # Generate response
            response = await llm.generate(
                prompt=request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            # Calculate final stats
            end_time = time.time()
            elapsed = end_time - start_time

            # Use actual usage if available, otherwise estimate
            # Safely get content length (handle None content)
            content_length = len(response.content) if response.content else 0

            if response.usage and isinstance(response.usage, dict):
                # Get actual tokens from usage, with fallback to estimates
                actual_input_tokens = response.usage.get("prompt_tokens")
                if actual_input_tokens is None or not isinstance(
                    actual_input_tokens, (int, float)
                ):
                    actual_input_tokens = input_tokens

                actual_output_tokens = response.usage.get("completion_tokens")
                if actual_output_tokens is None or not isinstance(
                    actual_output_tokens, (int, float)
                ):
                    # Fallback to estimation based on content length
                    actual_output_tokens = content_length // 4
            else:
                # No usage info, estimate from content
                actual_input_tokens = input_tokens
                actual_output_tokens = content_length // 4

            # Final validation - ensure we have valid numbers (not None)
            if actual_input_tokens is None:
                actual_input_tokens = input_tokens
            if actual_output_tokens is None:
                actual_output_tokens = content_length // 4

            # Convert to int to ensure we have integers (handle None, 0, or any numeric value)
            try:
                actual_input_tokens = (
                    int(actual_input_tokens)
                    if actual_input_tokens is not None
                    else input_tokens
                )
            except (TypeError, ValueError):
                actual_input_tokens = input_tokens

            try:
                actual_output_tokens = (
                    int(actual_output_tokens) if actual_output_tokens is not None else 0
                )
            except (TypeError, ValueError):
                actual_output_tokens = content_length // 4

            # Final safety check - ensure we have valid integers, never None
            if actual_input_tokens is None or not isinstance(
                actual_input_tokens, (int, float)
            ):
                actual_input_tokens = input_tokens
            if actual_output_tokens is None or not isinstance(
                actual_output_tokens, (int, float)
            ):
                actual_output_tokens = content_length // 4

            # Convert to int (final conversion)
            actual_input_tokens = int(actual_input_tokens)
            actual_output_tokens = int(actual_output_tokens)

            # Calculate tokens per second safely
            # At this point, actual_output_tokens MUST be an int >= 0
            if elapsed > 0 and actual_output_tokens > 0:
                tokens_per_second = actual_output_tokens / elapsed
            else:
                tokens_per_second = 0.0

            # Send final stats
            await broadcast_stats(
                {
                    "request_id": request_id,
                    "provider": request.provider,
                    "input_tokens": actual_input_tokens,
                    "output_tokens": actual_output_tokens,
                    "tokens_per_second": round(tokens_per_second, 2),
                    "time_to_first_token": round(
                        elapsed, 3
                    ),  # For non-streaming, same as total time
                    "elapsed_time": round(elapsed, 3),
                    "status": "completed",
                }
            )

            # Update state to healthy if generation succeeds
            llm._set_state(LLMState.HEALTHY)
            # Note: Status updates will be pushed via WebSocket automatically
            return {
                "content": response.content,
                "model": response.model,
                "provider": request.provider,
                "usage": response.usage,
                "metadata": response.metadata,
            }
    except HTTPException:
        raise
    except Exception as e:
        # Update state to unhealthy on error
        llm._set_state(LLMState.UNHEALTHY, str(e))
        logger.error(
            f"Error generating response with {request.provider}: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )


@router.post("/stream")
async def stream_text(request: GenerateRequest):
    """Stream text generation from the LLM"""
    # Get the LLM instance for the specified provider
    llm = get_llm_instance(request.provider)

    logger.info(
        f"Streaming text - provider: {request.provider}, "
        f"temperature: {request.temperature}, prompt: {request.prompt[:50]}..."
    )

    async def generate_stream():
        try:
            async for chunk in llm.generate_stream(
                prompt=request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(
                f"Error streaming response with {request.provider}: {str(e)}",
                exc_info=True,
            )
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.websocket("/ws/status")
async def websocket_status_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for pushing health/status updates and generation stats

    Server automatically pushes updates:
    - On connection: Initial status for all providers
    - Periodically: Status updates every 30 seconds
    - On state change: When provider state changes
    - During generation: Real-time stats (tokens/sec, input tokens, time to first token)

    Message format (Server -> Client):
    - {"type": "status", "status": "...", "providers": {...}} - Overall status
    - {"type": "status_update", "provider": "...", "state": "...", "model": "...", "error_message": "..."} - Provider update
    - {"type": "generation_stats", "request_id": "...", "input_tokens": 10, "output_tokens": 50, "tokens_per_second": 25.5, "time_to_first_token": 0.5, "elapsed_time": 2.0} - Generation stats
    """
    await websocket.accept()
    logger.info("WebSocket status connection established")

    # Add to stats WebSocket set
    stats_websockets.add(websocket)

    # Send initial status on connection
    try:
        await push_all_providers_status(websocket)
    except Exception as e:
        logger.error(f"Failed to send initial status: {e}", exc_info=True)

    # Track previous states to detect changes
    previous_states = {}
    for prov, llm_instance in llm_instances.items():
        previous_states[prov] = llm_instance.state.value

    try:
        # Set up periodic status updates
        update_interval = 30  # seconds - check for state changes

        while True:
            # Wait for update interval
            await asyncio.sleep(update_interval)

            # Check for state changes
            state_changed = False
            for prov, llm_instance in llm_instances.items():
                current_state = llm_instance.validate_config().value
                if previous_states.get(prov) != current_state:
                    state_changed = True
                    previous_states[prov] = current_state
                    # Push individual provider update
                    try:
                        # Check if WebSocket is still connected
                        if (
                            hasattr(websocket, "client_state")
                            and websocket.client_state.name != "CONNECTED"
                        ):
                            break

                        update_data = {
                            "type": "status_update",
                            "provider": prov,
                            "state": current_state,
                            "model": llm_instance.model,
                        }
                        # Only include error_message if state is not healthy
                        if (
                            current_state != LLMState.HEALTHY.value
                            and llm_instance.error_message
                        ):
                            update_data["error_message"] = llm_instance.error_message
                        await websocket.send_json(update_data)
                    except (RuntimeError, ConnectionError, WebSocketDisconnect) as e:
                        logger.debug(
                            f"WebSocket disconnected while pushing status update for {prov}: {e}"
                        )
                        break
                    except Exception as e:
                        error_msg = str(e).lower()
                        if any(
                            keyword in error_msg
                            for keyword in ["close", "closed", "disconnect", "send"]
                        ):
                            logger.debug(
                                f"Connection error pushing status update for {prov}: {e}"
                            )
                            break
                        else:
                            logger.error(
                                f"Failed to push status update for {prov}: {e}"
                            )
                            break

            # Push full status update on state change
            if state_changed:
                try:
                    # Check if WebSocket is still connected
                    if (
                        hasattr(websocket, "client_state")
                        and websocket.client_state.name != "CONNECTED"
                    ):
                        break
                    await push_all_providers_status(websocket)
                except (RuntimeError, ConnectionError, WebSocketDisconnect) as e:
                    logger.debug(
                        f"WebSocket disconnected while pushing status update: {e}"
                    )
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(
                        keyword in error_msg
                        for keyword in ["close", "closed", "disconnect", "send"]
                    ):
                        logger.debug(f"Connection error pushing status update: {e}")
                        break
                    else:
                        logger.error(f"Failed to push status update: {e}")
                        break

    except WebSocketDisconnect:
        logger.info("WebSocket status client disconnected")
        stats_websockets.discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket status error: {str(e)}", exc_info=True)
        stats_websockets.discard(websocket)
    finally:
        try:
            stats_websockets.discard(websocket)
            # Check if WebSocket is still open before closing
            if (
                hasattr(websocket, "client_state")
                and websocket.client_state.name == "CONNECTED"
            ):
                await websocket.close()
        except (RuntimeError, ConnectionError, WebSocketDisconnect, OSError) as e:
            # Ignore errors during shutdown - socket may already be closed
            error_msg = str(e).lower()
            if "bad file descriptor" not in error_msg:
                logger.debug(f"WebSocket cleanup: {e}")
        except Exception as e:
            # Log unexpected errors but don't fail
            logger.debug(f"WebSocket cleanup error: {e}")


async def close_all_websockets():
    """Close all WebSocket connections gracefully during shutdown"""
    if not stats_websockets:
        return

    logger.info(f"Closing {len(stats_websockets)} WebSocket connection(s)")
    closed = set()

    for websocket in list(stats_websockets):
        try:
            # Check if WebSocket is still connected
            if (
                hasattr(websocket, "client_state")
                and websocket.client_state.name == "CONNECTED"
            ):
                await websocket.close()
            closed.add(websocket)
        except (RuntimeError, ConnectionError, WebSocketDisconnect, OSError) as e:
            # Ignore errors during shutdown - socket may already be closed
            error_msg = str(e).lower()
            if "bad file descriptor" not in error_msg:
                logger.debug(f"Error closing WebSocket during shutdown: {e}")
            closed.add(websocket)
        except Exception as e:
            logger.debug(f"Unexpected error closing WebSocket: {e}")
            closed.add(websocket)

    # Remove all closed WebSockets
    stats_websockets.difference_update(closed)
    logger.info(f"Closed {len(closed)} WebSocket connection(s)")


async def perform_health_checks_for_all_providers():
    """Perform health checks for all configured providers"""

    async def check_provider(prov: str, llm_instance: BaseLLM):
        """Check health for a single provider"""
        try:
            if llm_instance.is_configured() and llm_instance.state != LLMState.HEALTHY:
                # Only check if not already healthy
                await llm_instance.check_health()
                logger.info(
                    f"Health check completed for {prov}: {llm_instance.state.value}"
                )
        except Exception as e:
            logger.error(f"Health check failed for {prov}: {e}", exc_info=True)

    # Perform health checks concurrently for all providers
    tasks = [
        check_provider(prov, llm_instance)
        for prov, llm_instance in llm_instances.items()
        if llm_instance.is_configured()
    ]

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def push_all_providers_status(websocket: WebSocket):
    """Push status for all providers"""
    # Check if WebSocket is still connected
    if (
        hasattr(websocket, "client_state")
        and websocket.client_state.name != "CONNECTED"
    ):
        return

    health_status = {}
    for prov, llm_instance in llm_instances.items():
        state = llm_instance.validate_config()
        # Only include error_message if state is not healthy
        provider_data = {
            "state": state.value,
            "model": llm_instance.model,
        }
        if state != LLMState.HEALTHY and llm_instance.error_message:
            provider_data["error_message"] = llm_instance.error_message
        health_status[prov] = provider_data

    # Determine overall status
    states = [h["state"] for h in health_status.values()]
    if all(s == LLMState.HEALTHY.value for s in states):
        overall_status = "healthy"
    elif any(s == LLMState.HEALTHY.value for s in states):
        overall_status = "partial"
    elif any(s == LLMState.UNHEALTHY.value for s in states):
        overall_status = "unhealthy"
    else:
        overall_status = "not_configured"

    try:
        await websocket.send_json(
            {
                "type": "status",
                "status": overall_status,
                "providers": health_status,
            }
        )
    except (RuntimeError, ConnectionError, WebSocketDisconnect) as e:
        logger.debug(f"WebSocket disconnected while pushing provider status: {e}")
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if any(
            keyword in error_msg
            for keyword in ["close", "closed", "disconnect", "send"]
        ):
            logger.debug(f"Connection error pushing provider status: {e}")
            raise
        else:
            logger.error(f"Failed to push provider status: {e}")
            raise
