#!/bin/bash

#export DASHSCOPE_API_KEY=''
export MODEL_SERVICE_URL='http://xxx.xxx.xxx.xxx:xxx/api/'
export FEEDBACK_SERVICE_URL='http://xxx.xxx.xxx.xxx:xxx/api/feedback'

python ./studio/gradio_with_feedback.py
