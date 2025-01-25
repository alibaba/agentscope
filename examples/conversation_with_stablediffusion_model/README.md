# Conversation with Stable-diffusion model

This example will show

- How to use Stable Diffusion models in AgentScope.

In this example, you can interact in a conversational format to generate images.
Once the image is generated, the agent will respond with the local file path where the image is saved.

## Minimum Hardware Requirements

- **GPU**: NVIDIA GPU with at least 6.9GB of VRAM
- **CPU**: Modern multi-core CPU (e.g., Intel i5 or AMD Ryzen 5)
- **RAM**: Minimum 8GB
- **Storage**: At least 10GB of available hard drive space

## How to Run

You need to satisfy the following requirements to run this example:

### Step 0: Install Stable Diffusion Web UI and AgentScope

- Install Stable Diffusion Web UI by following the instructions at [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- Install the latest version of AgentScope by
  ```bash
  git clone https://github.com/modelscope/agentscope.git
  cd agentscope
  pip install -e .
  ```

### Step 1: Download the required checkpoints

Before starting the Stable Diffusion Web UI, you need to download at least one model to ensure normal operation.
Download the model to `stable-diffusion-webui/models/Stable-diffusion` directory.

### Step 2: Launch the Stable Diffusion Web UI

We've provided a convenient shell script to quickly start the Stable Diffusion Web UI:
`scripts/stable_diffusion_webui/sd_setup.sh`

Activate the virtual environment first, Then, run the following command in your terminal, replacing YOUR-SD-WEBUI-PATH with the actual path to your Stable Diffusion Web UI directory:

```bash
bash scripts/stable_diffusion_webui/sd_setup.sh -s YOUR-SD-WEBUI-PATH
```

If you choose to start it on your own, you need to launch the Stable Diffusion Web UI with the following arguments: `--api --port=7862`. For more detailed instructions on starting the WebUI, refer to the [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

### Step 3: Running the Example

Run the example and input your prompt.

```bash
python conversation_with_stablediffusion_model.py
```

## Customization Options

### `model_config` Example:

```json
{
  "model_type": "sd_txt2img",
  "config_name": "sd",
  "options": {
    "sd_model_checkpoint": "Anything-V3.0-pruned",
    "sd_lora": "add_detail",
    "CLIP_stop_at_last_layers": 2
  },
  "generate_args": {
    "steps": 50,
    "n_iter": 1,
    "override_settings": {
      "CLIP_stop_at_last_layers": 3
    }
  }
}
```

### Parameter Explanation:

- `options`: Global configuration that directly affects the WebUI settings.
- `generate_args`: Controls parameters for individual image generation requests, such as `steps` (number of sampling steps) and `n_iter` (number of iterations).
  - `override_settings`: Overrides WebUI settings for a single request, taking precedence over `options`.

Notes:

- `override_settings` only affects the current request, while changes made to `options` persist.
- Both parameters can set the same options, but `override_settings` has a higher priority.

As shown in the example, the final image will be generated with the following settings:

steps: 50
n_iter: 1
sd_model_checkpoint: Anything-V3.0-pruned
sd_lora: add_detail
CLIP_stop_at_last_layers: 3

However, the web UI will always display the following settings:

sd_model_checkpoint: Anything-V3.0-pruned
sd_lora: add_detail
CLIP_stop_at_last_layers: 2

### Available Parameter Lists:

If you've successfully enabled the Stable Diffusion Web UI API, you should be able to access its documentation at http://127.0.0.1:7862/docs (or whatever URL you're using + /docs).

- `generate_args`: {url}/docs#/default/text2imgapi_sdapi_v1_txt2img_post
- `options` and `override_settings`: {url}/docs#/default/get_config_sdapi_v1_options_get

For this project, the "options" parameter will be posted to the /sdapi/v1/options API endpoint,
and the "generate_args" parameter will be posted to the /sdapi/v1/txt2img API endpoint.
You can refer to https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API for a more parameter reference guide.

## A Running Example

- Conversation history with Stable Diffusion Web UI.
  ```bash
  User input:Horses on Mars
  User: Horses on Mars
  Assistant: Image saved to path\agentscope\runs\run_20240920-142208_rqsvhh\file\image_20240920-142522_HTF38X.png
  User input: boy eating ice-cream
  User: boy eating ice-cream
  Assistant: Image saved to path\agentscope\runs\run_20240920-142208_rqsvhh\file\image_20240920-142559_2xGtUs.png
  ```
- Image
<img src="https://img.alicdn.com/imgextra/i3/O1CN01YoMRQP26ClOHM7Kh0_!!6000000007626-0-tps-512-512.jpg" alt="Horses on Mars" width="300" />
<img src="https://img.alicdn.com/imgextra/i1/O1CN01QTO8AU1HVxaQ2rFPx_!!6000000000764-0-tps-512-512.jpg" alt="boy eating ice-cream" width="300" />