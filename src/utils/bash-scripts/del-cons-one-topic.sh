#!/bin/bash

# Usage:
# ./delete_groups_for_topic.sh topicname

topic="$1"

if [[ -z "$topic" ]]; then
    echo "ERROR: No topic specified."
    exit 1
fi

echo "Deleting consumer groups subscribed to topic: $topic"

for g in $(rpk group list | awk '{print $1}'); do
    if rpk group describe "$g" | grep -q "$topic"; then
        echo "Deleting group: $g"
        rpk group delete "$g"
    fi
done

echo "Done."
