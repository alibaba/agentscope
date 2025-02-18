# KIMAs: A Configurable Knowledge Integrated Multi-Agent System

## Introduction
KIMAs stands for <ins>k</ins>nowledge <ins>i</ins>ntegrated <ins>m</ins>ulti-<ins>a</ins>gent <ins>s</ins>ystem, with configurable components and the capability of utilizing multi-source knowledge. It features a flexible and configurable system for integrating diverse knowledge sources with 1) context management and query rewrite mechanisms to improve retrieval accuracy and multi-turn conversational coherency, 2) efficient knowledge routing and retrieval, 3) simple but effective filter and reference generation mechanisms, and 4) optimized parallelizable multi-agent pipeline execution.

For more details about the design, please refer to [this paper](https://arxiv.org/abs/2502.09596). If you want to take a try, welcome to join the following AgentScope QA DingTalk group and ask the bot by "@AgentScope答疑机器人 {YOUR QUESTION}".

<img src="https://img.alicdn.com/imgextra/i2/O1CN01tuJ5971OmAqNg9cOw_!!6000000001747-0-tps-444-460.jpg" width="100" height="100">

## Key components
* [copilot_app.py](copilot_app.py): The core of the copilot application, including main logistics of the application.
* [copilot_server.py](copilot_server.py): Setup a server that load local `copilot_core` pipeline, or remote dashscope pipeline, and connect to the client such as DingTalk/Gradio

⚠️ This application relies on asynchronous mode of DashScope API. Therefore, all the model API calls are through DashScope library. Other library/model APIs may not be compatible.

## Pre-runing settings

Create a conda environment:
`conda create -n "copilot_core" python=3.9`

Install the latest AgentScope:
```bash
git clone git@github.com:modelscope/agentscope.git
cd agentscope
pip install -e .\[full\]
```

Change directory and install application-specific packages:
```bash
cd application/multisource_rag_app
pip install -r requirements.txt
```

Environment variables and API keys:
* setup the API keys in the terminal or `.sh` files.
  ````bash
  export DASHSCOPE_API_KEY='your_dash_key'
  export BING_SEARCH_KEY='your_bing_key' # optional, only required if you have bing knowledge
  ````


## Setup a RAG application service
#### Ready to run:
```bash
cd src
as_scripts/setup_server.sh
```

There are some variables that are already set in the script. If any developers want to migrate the service for their own applications, the following description can be used as a reference.

####  Configurable variables in the [server setup script](as_scripts/setup_server.sh):
* Environment variable `DASHSCOPE_API_KEY`: The key of the dashscope service.
* (Optional) Environment variable `BING_SEARCH_KEY`: The key of Bing search if the application uses Bing search (`BingKnowledge`).
* Environment variable `RAG_AGENT_NAMES`: Names of the retrieval agents, seperated by ",".
* Environment variable `RAG_RETURN_RAW`: Whether the retrieval agents only return raw content (Recommend and default to be `True`).
* Environment variable `MODEL_CONFIG_PATH` : Path to the AgentScope model configurations.
* Environment variable `AGENT_CONFIG_PATH`: Path to the agent configurations.
* Environment variable `KNOWLEDGE_CONFIG_PATH`: Path to the knowledge configuration.
* Environment variable `SERVER_MODEL`: It determines the working mode of the server, you may choose "dash" (remote service on dashscope), "local" (local service) or "dummy" (testing with dummy response).
* Environment variable `DB_PATH`: It determines the saving path of the sql tables, if not set, the default path is under the ```runs``` folder.

* Server setup variables after `uvicorn`:
  * `--port`: This port is used for the copilot server, corresponding gradio server would communicate via this port
  * `--host 0.0.0.0`: Listen on all available network interfaces of the machine.
  * `--loop asyncio`: Ask Uvicorn to use Python’s built-in asyncio event loop.


## Run a Gradio interface

A Gradio interface can be set up simply by the following script.

```bash
as_scripts/setup_gradio.sh
```

If any developers want to setup the service for their own applications, the following description can be used as a reference.

####  Configurable variables in the [Gradio setup script](as_scripts/setup_gradio.sh)
* Environment variable `DASHSCOPE_API_KEY`: The key of the dashscope service.
* `MODEL_SERVICE_URL`: The url of the RAG application service, by default is `http://xx.xx.xx.xx:xxxx/api`
* `FEEDBACK_SERVICE_URL`: The url of RAG application service accepting portal, by default it is `http://xx.xx.xx.xx:xxxx/api/feedback`


## (Optional) Using ElasticSearch
Loading data from disk can be very time consuming if the data volume is large. A more standard way is to use a vector database to host the data and the computed embeddings.

* To use the ElasticSearch, you first need to download the [ElasticSearch](https://www.elastic.co/downloads/elasticsearch).

* If running for the AgentScope QA, check the `input_dir` files in `./configs/as_config/as_knowledge_configs/as_es_knowledge_update_config.json`.
  * The one of `agentscope_tutorial_rag` should be `{PATH_TO_STORE_TUTORIAL_TXT_FILES}`
  * Others `input_dir` should be consistent with where the AgentScope materials are.

* You need run the following command to prepare the data under the `multisource_rag_app/src`
  ```bash
  REPO_PATH={PATH_TO_AGENTSCOPE_REPO} TEXT_DIR={PATH_TO_STORE_TUTORIAL_TXT_FILES} python prepare_knowledge.py -r as
  ```

* You need to change the `KNOWLEDGE_CONFIG_PATH` in server setup script to `./configs/as_config/as_knowledge_configs/as_es_knowledge_read_config.json`
