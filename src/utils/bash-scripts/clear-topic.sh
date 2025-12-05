#!/bin/bash

# Usage:
# ./clean_topic.sh topicname

topic="$1"

if [[ -z "$topic" ]]; then
    echo "ERROR: No topic specified."
    exit 1
fi

echo "Cleaning topic: $topic"

rpk topic delete "$topic"
sleep 2
rpk topic create "$topic"

echo "Done."
