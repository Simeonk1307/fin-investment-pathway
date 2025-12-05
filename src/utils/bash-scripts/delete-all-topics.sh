#!/bin/bash

echo "Deleting ALL topics in the cluster..."

topics=$(rpk topic list | awk '{print $1}')

for t in $topics; do
    echo "Deleting topic: $t"
    rpk topic delete "$t"
done

echo "Done."
