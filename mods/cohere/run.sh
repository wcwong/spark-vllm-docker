#!/bin/bash

set -e

uv pip install cohere_melody
cp chat_template.jinja $WORKSPACE_DIR/

