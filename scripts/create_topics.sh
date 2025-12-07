#!/bin/bash

BROKER="redpanda-0:9092"

echo "⏳ Waiting for cluster..."
until [ $(rpk cluster info --brokers $BROKER 2>/dev/null | grep -c "redpanda-") -ge 3 ]; do
  sleep 5
done
sleep 10

echo "✅ Creating topics..."

TOPICS=(
  "bronze.news:6:3:cleanup.policy=delete,retention.ms=604800000"
  "bronze.socials:6:3:cleanup.policy=delete,retention.ms=604800000"
  "bronze.stocks:6:3:cleanup.policy=delete,retention.ms=604800000"
  "bronze.filings:6:3:cleanup.policy=delete,retention.ms=604800000"
  "silver.news:6:3:cleanup.policy=compact"
  "silver.socials:6:3:cleanup.policy=compact"
  "silver.stocks:6:3:cleanup.policy=compact"
  "silver.filings:6:3:cleanup.policy=compact"
  "silver.dlq.news:3:3:cleanup.policy=delete,retention.ms=1209600000"
  "silver.dlq.socials:3:3:cleanup.policy=delete,retention.ms=1209600000"
  "silver.dlq.stocks:3:3:cleanup.policy=delete,retention.ms=1209600000"
  "silver.dlq.filings:3:3:cleanup.policy=delete,retention.ms=1209600000"
)

for entry in "${TOPICS[@]}"; do
  IFS=':' read -r name partitions replicas config <<< "$entry"
  
  for attempt in 1 2 3; do
    if rpk topic create "$name" --brokers $BROKER -p $partitions -r $replicas --topic-config ${config//,/ --topic-config } 2>&1 | grep -q "OK\|exists"; then
      echo "✓ $name"
      break
    fi
    echo "  Retry $attempt for $name..."
    sleep 2
  done
done

echo ""
echo "📋 Topics:"
rpk topic list --brokers $BROKER