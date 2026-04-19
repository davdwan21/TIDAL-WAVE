#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URL="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Read from Terraform outputs
cd "$(dirname "$0")/infra"
REPO=$(terraform output -raw ecr_repository_url)
CLUSTER=$(terraform output -raw ecs_cluster)
TASK_DEF=$(terraform output -raw task_definition_arn)
SUBNETS=$(terraform output -json subnet_ids 2>/dev/null || true)
cd ..

echo "=== Building image ==="
docker build -t simulation ./project

echo "=== Pushing to ECR ==="
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URL"
docker tag simulation:latest "${REPO}:latest"
docker push "${REPO}:latest"

echo "=== Running Fargate task ==="
# Edit SUBNETS and SECURITY_GROUPS below if not using Terraform outputs
SUBNET="${1:-}"
SG="${2:-}"

if [[ -z "$SUBNET" || -z "$SG" ]]; then
  echo "Usage: ./deploy.sh <subnet-id> <security-group-id>"
  echo "  e.g. ./deploy.sh subnet-abc123 sg-def456"
  exit 1
fi

aws ecs run-task \
  --cluster "$CLUSTER" \
  --task-definition "$TASK_DEF" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNET}],securityGroups=[${SG}],assignPublicIp=ENABLED}" \
  --region "$REGION"

echo "Task launched. Logs: https://console.aws.amazon.com/cloudwatch/home?region=${REGION}#logsV2:log-groups/log-group/%2Fecs%2F$(terraform -chdir=infra output -raw ecs_cluster 2>/dev/null || echo datahacks-simulation)"
