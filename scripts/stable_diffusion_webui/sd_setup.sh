#!/bin/bash

# set VENV_DIR=%~dp0%venv
# call "%VENV_DIR%\Scripts\activate.bat"

# stable_diffusion_webui_path="YOUR_PATH_TO_STABLE_DIFFUSION_WEBUI"

port=7862

while getopts ":p:s:" opt
do
    # shellcheck disable=SC2220
    case $opt in
        p) port="$OPTARG";;
        s) stable_diffusion_webui_path="$OPTARG"
        ;;
    esac
done

stable_diffusion_webui_path=${stable_diffusion_webui_path%/}
launch_py_path="$stable_diffusion_webui_path/launch.py"

# Check if the launch.py script exists
if [[ ! -f "$launch_py_path" ]]; then
    echo "The launch.py script was not found at $launch_py_path."
    echo "Please ensure you have specified the correct path to your Stable Diffusion WebUI using the -s option."
    echo "Example: ./sd_setup.sh -s /path/to/your/stable-diffusion-webui"
    echo "Alternatively, you can set the path directly in the script."
    exit 1
fi

cd $stable_diffusion_webui_path

python ./launch.py --api --port=$port
