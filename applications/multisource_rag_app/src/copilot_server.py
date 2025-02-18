# -*- coding: utf-8 -*-
# pylint: disable=E0611
"""
Set up a server for the rag application
"""
import json
import os
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Generator

import dashscope
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from flask_httpauth import HTTPTokenAuth

from copilot_app import copilot_bot_pipeline, initializations
from utils.db_logging import (
    create_logging_table,
    record_user_input,
    record_model_response,
    record_suggest_answer,
    record_score,
)
from utils.logging import logger

app = FastAPI()
auth = HTTPTokenAuth(scheme="Bearer")
DB_NAME = os.getenv(
    "DB_PATH",
    f'logs/runs-{datetime.now().strftime("%Y-%m-%d-%H-%M")}.db',
)

# support three kind of model deployment modes: local, dash, dummy
# local: locally run pipeline to obtain response
# dash: call dash server to obtain the response
# dummy: for testing, generate compatible dummy output
SERVER_MODEL = os.getenv(
    "SERVER_MODEL",
    "dummy",
)
assert SERVER_MODEL in ["local", "dash", "dummy"]
if SERVER_MODEL == "dummy":
    logger.warning(
        "Running the server with dummy model; will not generate m"
        "eaningful results.",
    )
if SERVER_MODEL == "local":
    initializations()

# create table for logging
create_logging_table(DB_NAME)


# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],  # Adjust to specific domains in production, e.g. ["http://example.com"]
    allow_credentials=True,
    allow_methods=[
        "*",
    ],  # Allow all HTTP methods, or specify a list,
    allow_headers=[
        "*",
    ],  # Allow all headers, or specify allowed headers
)

# Define your API keys
API_KEYS = {
    os.getenv("DASHSCOPE_API_KEY", ""): "real",
}


@auth.verify_token
def verify_token(token: str) -> Any:
    """verify a token"""
    if token in API_KEYS:
        return API_KEYS[token]
    return None


# internal api key verification
def require_api_key(f: Any) -> Any:
    """wrapper for api key validation"""

    @wraps(f)
    async def decorated(request: Request, *args: Any, **kwargs: Any) -> Any:
        token = request.headers.get("authorization", "")
        if token:
            token = token.split(" ")[1]  # Remove 'Bearer ' prefix
        else:
            request_id = uuid.uuid4()
            return_json = {
                "request_id": str(request_id),
                "code": "InvalidApiKey",
                "message": "The API key in your request is missing.",
            }
            return StreamingResponse(
                f"id: {request_id}\nevent:error\nstatus: 401\ndata: "
                f"{json.dumps(return_json, ensure_ascii=False)}\n\n",
                media_type="text/event-stream",
            )

        if not verify_token(token):
            request_id = uuid.uuid4()
            return_json = {
                "request_id": str(request_id),
                "code": "InvalidApiKey",
                "message": "Invalid API key.",
            }

            return StreamingResponse(
                f"id: {id}\nevent:error\nstatus: 401\ndata:  "
                f"{json.dumps(return_json, ensure_ascii=False)}\n\n",
                media_type="text/event-stream",
            )

        return await f(request, *args, **kwargs)

    return decorated


def model_response(
    request_id: str,
    data: dict,
    **kwargs: Any,  # noqa: W0613
) -> Any:
    # pylint: disable=unused-argument
    """generate model response"""
    messages = data["input"]["messages"]
    if SERVER_MODEL == "dash":
        dashscope.base_http_api_url = (
            "https://poc-dashscope.aliyuncs.com/api/v1"
        )
        custom = {
            "scene": "rag_chat",
        }
        responses = dashscope.Generation.call(
            model=os.getenv("TEST_MODEL", "pre-ms_copilot_test-2349"),
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            messages=messages,
            result_format="messages",  # 将返回结果格式设置为 message
            stream=True,
            custom=custom,
        )
    elif SERVER_MODEL == "local":
        # TODO: to be tested (@zitao)
        # from copilot_app import copilot_bot_pipeline
        responses = copilot_bot_pipeline(
            request_payload=data,
            request_id=request_id,
        )
    else:

        def dummy_output_generator() -> Generator:
            dummy_output = "test server with dummy output"
            for i in range(len(dummy_output)):
                output = {
                    "output": {
                        "choices": [
                            {
                                "messages": [
                                    {
                                        "role": "assistant",
                                        "name": "assistant",
                                        "from": "text",
                                        "msg_id": request_id,
                                        "content": dummy_output[:i],
                                    },
                                ],
                            },
                        ],
                    },
                    "usage": {},
                }
                yield output

        responses = dummy_output_generator()
    return responses


def stream_response(request_id: str, data: dict, **kwargs: Any) -> Any:
    """generate stream response"""
    output_package = {}
    for output_package in model_response(request_id, data, **kwargs):
        output_package["request_id"] = request_id
        yield (
            f"id: {request_id}\nevent: result\nstatus: 200\ndata: "
            f"{json.dumps(output_package, ensure_ascii=False)}\n\n"
        )
    record_model_response(
        DB_NAME,
        request_id,
        json.dumps(output_package, ensure_ascii=False),
    )


def complete_response(request_id: str, data: dict, **kwargs: Any) -> Any:
    """generate complete response"""
    final_response = {}
    for response in model_response(request_id, data, **kwargs):
        final_response = response

    http_return = {
        "success": True,
    }
    http_return.update(final_response)
    record_model_response(
        DB_NAME,
        request_id,
        json.dumps(final_response, ensure_ascii=False),
    )
    return http_return


@app.post("/api/services/aigc/text-generation/generation")
@require_api_key
async def generate(request: Request) -> Any:
    """Async generation api"""
    data = await request.json()
    logger.info(str(data))
    request_id = str(uuid.uuid4()) + str(
        datetime.now().strftime("%m%d%Y%H%M%S"),
    )
    record_user_input(
        DB_NAME,
        request_id,
        json.dumps(
            data.get("input", {}).get("messages", []),
            ensure_ascii=False,
        ),
    )
    is_stream = request.headers.get("x-dashscope-sse", False)

    if is_stream:
        return StreamingResponse(
            stream_response(request_id, data),
            media_type="text/event-stream",
        )
    else:
        return complete_response(request_id, data)


@app.post("/api/feedback")
@require_api_key
async def feedback(request: Request) -> Any:
    """Async feedback api"""
    data = await request.json()
    logger.info(data)
    request_id = data.get("id", "")
    if "score" in data:
        score = float(data["score"])
        logger.info(f"logging score for {request_id}: {score}")
        record_score(DB_NAME, request_id, score)
    if "suggest_answer" in data:
        suggest_answer = data["suggest_answer"]
        logger.info(
            f"logging suggest_answer for {request_id}: {suggest_answer}",
        )
        record_suggest_answer(DB_NAME, request_id, suggest_answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5006, loop="asyncio")
