#!/bin/bash
set -e

echo "=========================================="
echo "Starting SASL Bootstrap Process"
echo "=========================================="
echo "RP_ENABLE_SASL: ${RP_ENABLE_SASL}"
echo "RP_SUPERUSER: ${RP_SUPERUSER}"

if [ "$RP_ENABLE_SASL" != "true" ]; then
  echo "SASL is disabled, skipping authentication setup"
  exit 0
fi

echo "Waiting for Redpanda cluster to stabilize..."
sleep 15

echo "Creating SASL user '${RP_SUPERUSER}'..."
rpk acl user create "${RP_SUPERUSER}" \
  --password "${RP_SUPERPASS}" \
  --api-urls redpanda-0:9644 || echo "User may already exist"

echo "Granting ACL permissions..."

rpk acl create \
  --allow-principal "User:${RP_SUPERUSER}" \
  --operation all \
  --topic "*" \
  --api-urls redpanda-0:9644 || echo "ACL may already exist"

rpk acl create \
  --allow-principal "User:${RP_SUPERUSER}" \
  --operation all \
  --group "*" \
  --api-urls redpanda-0:9644 || echo "ACL may already exist"

rpk acl create \
  --allow-principal "User:${RP_SUPERUSER}" \
  --operation all \
  --cluster \
  --api-urls redpanda-0:9644 || echo "ACL may already exist"

echo "Verifying SASL configuration..."
rpk acl user list --api-urls redpanda-0:9644

echo "=========================================="
echo "SASL Bootstrap Completed Successfully!"
echo "=========================================="
