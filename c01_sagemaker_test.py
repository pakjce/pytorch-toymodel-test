#!/usr/bin/env python3
import json
import argparse
import boto3
from sagemaker.predictor import RealTimePredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sagemaker Predictor'
    )
    parser.add_argument(
        '--endpoint', type=str, required=True,
        help='Sagemaker console 상의 Endpoint의 Name 속성'
    )
    parser.add_argument(
        '--file', type=str, required=True,
        help='Inference를 수행할 파일'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sm = boto3.client('sagemaker')

    endpoint_desc = sm.describe_endpoint(EndpointName=args.endpoint)

    print(endpoint_desc)
    print('----------')

    predictor = RealTimePredictor(
        endpoint=args.endpoint
    )

    with open(args.file, 'rb') as f:
        payload = f.read()
        payload = payload

    response = predictor.predict(data=payload).decode('utf-8')
    print(response)


if __name__ == '__main__':
    main()
