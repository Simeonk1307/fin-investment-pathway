#!/bin/bash

# Usage:
# ./clean_topics.sh topic1 topic2 topic3

for topic in "$@"; do
    echo "Cleaning topic: $topic"
    rpk topic delete "$topic"
    sleep 2
    rpk topic create "$topic"
done

echo "Done."
