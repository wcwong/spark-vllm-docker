#!/bin/bash
# https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates
# v19
set -e
cp chat_template.jinja $WORKSPACE_DIR/fixed_chat_template.jinja
echo "=======> to apply chat template, use --chat-template fixed_chat_template.jinja"
