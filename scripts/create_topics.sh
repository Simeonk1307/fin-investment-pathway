#!/bin/bash
set -e

echo "=========================================="
echo "Creating Redpanda Topics"
echo "=========================================="

AUTH_ARGS=""
if [ "$RP_ENABLE_SASL" = "true" ]; then
  AUTH_ARGS="--user ${RP_SUPERUSER} --password ${RP_SUPERPASS} --sasl-mechanism ${RP_SASL_MECHANISM}"
  echo "Using SASL authentication"
else
  echo "Using no authentication"
fi

create_topic() {
  local topic=$1
  local partitions=${2:-3}
  local replicas=${3:-3}
  local retention=${4:-604800000}
  
  echo "Creating topic: $topic (partitions=$partitions, replicas=$replicas)"
  
  rpk topic create "$topic" \
    --brokers ${RP_BROKERS} \
    --partitions $partitions \
    --replicas $replicas \
    --topic-config retention.ms=$retention \
    --topic-config compression.type=snappy \
    $AUTH_ARGS || echo "Topic '$topic' already exists"
}

echo ""
echo "Creating Bronze Layer Topics..."
create_topic "bronze.stocks" 3 3 604800000
create_topic "bronze.socials" 3 3 604800000
create_topic "bronze.news" 3 3 604800000
create_topic "bronze.filings" 3 3 604800000

echo ""
echo "Creating Silver Layer Topics..."
create_topic "silver.stocks" 3 3 2592000000
create_topic "silver.socials" 3 3 2592000000
create_topic "silver.news" 3 3 2592000000
create_topic "silver.filings" 3 3 2592000000

echo ""
echo "Creating Dead Letter Queue Topics..."
create_topic "silver.dlq.stocks" 1 3 2592000000
create_topic "silver.dlq.news" 1 3 2592000000
create_topic "silver.dlq.socials" 1 3 2592000000
create_topic "silver.dlq.filings" 1 3 2592000000

echo ""
echo "Listing all topics..."
rpk topic list --brokers ${RP_BROKERS} $AUTH_ARGS

echo ""
echo "=========================================="
echo "Topics Created Successfully!"
echo "=========================================="
