#!/bin/bash

# Pass topics as arguments:
# ./delete_topics.sh topic1 topic2 topic3

for t in "$@"; do
    echo "Deleting topic: $t"
    rpk topic delete "$t"
done

echo "Done."
