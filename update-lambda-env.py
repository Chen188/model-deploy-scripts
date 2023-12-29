import sys
import boto3
import time

if len(sys.argv) !=2:
    print("Usage: python3 update-lambda-env.py <region-name>")
    exit(1)

region=sys.argv[1]

with open('./bge-sagemaker-endpoint.txt', 'r') as f:
    bge_endpoint = f.read()


with open('./chatglm-sagemaker-endpoint.txt', 'r') as f:
    chatglm_endpoint = f.read()

_lambda = boto3.client('lambda', region_name=region)

def change_lambda_env(lambda_name):
    # describe lambda function 'Ask_Assistant' config 
    response = _lambda.get_function_configuration(FunctionName=lambda_name)
    cur_envs = response['Environment']['Variables']

    cur_envs['llm_model_endpoint'] = chatglm_endpoint # default to ''
    cur_envs['embedding_endpoint'] = bge_endpoint # default to 'cohere.embed-multilingual-v3'

    # update lambda function 'Ask_Assistant' enviroment config
    _lambda.update_function_configuration(
        FunctionName='Ask_Assistant',
        Environment={
            'Variables': cur_envs
        },
    )

for name in ['Ask_Assistant', 'Trigger_Ingestion', 'Detect_Intention', 'Query_Rewrite', 'Chat_Agent']:
    print(f'update: lambda {name}')
    change_lambda_env(name)
    print(f'lambda {name} env updated')
    time.sleep(5) # wait 5s before another update, in case ResourceConflictException
