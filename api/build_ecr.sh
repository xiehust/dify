#!/bin/bash

TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

# Get the current region and write it to the backend .env file
region=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/placement/region)
# region=$(aws configure get region)
suffix="com"

if [[ "$region" == cn*  ]]; then
    suffix="com.cn"
fi

account=$(aws sts  get-caller-identity --query Account --output text)

VERSION=latest
inference_image=dify-api
inference_fullname=${account}.dkr.ecr.${region}.amazonaws.${suffix}/${inference_image}:${VERSION}

# If the repository doesn't exist in ECR, create it.
aws  ecr describe-repositories --repository-names "${inference_image}" --region ${region} || aws ecr create-repository --repository-name "${inference_image}" --region ${region}

if [ $? -ne 0 ]
then
    aws  ecr create-repository --repository-name "${inference_image}" --region ${region}
fi

# Get the login command from ECR and execute it directly
aws  ecr get-login-password --region $region | docker login --username AWS --password-stdin $account.dkr.ecr.$region.amazonaws.${suffix}

aws ecr set-repository-policy \
    --repository-name "${inference_image}" \
    --policy-text "file://ecr-policy.json" \
    --region ${region}

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build --build-arg VERSION=${VERSION} -t ${inference_image}:${VERSION}  -f Dockerfile . 

docker tag ${inference_image}:${VERSION} ${inference_fullname}

docker push ${inference_fullname}

echo "Image URI: ${inference_fullname}"