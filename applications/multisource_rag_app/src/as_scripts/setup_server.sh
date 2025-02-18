#!/bin/bash

export MODEL_CONFIG_PATH=./configs/model_config.json
export AGENT_CONFIG_PATH=./configs/as_config/as_agent_configs/agent_config_dict.json
export KNOWLEDGE_CONFIG_PATH=./configs/as_config/as_knowledge_configs/knowledge_config.json

export RAG_AGENT_NAMES='tutorial_agent,api_agent,example_agent'
export RAG_RETURN_RAW=True
# export DASHSCOPE_API_KEY=xxx

# you may choose "dash"(remote service), "local"(local service) or "dummy"(testing)
export SERVER_MODEL=local
# this is the path of the SQLlite database file, configure before running
export DB_PATH=./logs/log_as_$(date +%Y%m%d).db

mkdir -p logs

# Notice, setting --loop flag as below to compatible with ES async
uvicorn copilot_server:app --host 0.0.0.0 --port 6009 --loop asyncio
