import os
import sys
import time
import json

try:
    region=sys.argv[1]
    assert(len(region.split('-')) == 3)
except:
    print('Usage: python deploy-chatglm.py [region-name]')
    raise("You've to provide region code, e.g. us-west-2")

# install pip requirements
print('installing pip requirements...')
os.system('pip install sagemaker huggingface-hub -Uqq')

from utils import *

import boto3

session = boto3.Session(region_name=region)

s3 = session.client('s3')
iam = session.client('iam')

s3_client = session.client("s3")
sm_client = session.client("sagemaker")
smr_client = session.client("sagemaker-runtime")

cfn_outpus = get_cfn_output(region=region)
bucket = cfn_outpus.get('UPLOADBUCKET')
account_id = bucket.split('-')[0]

# download model
def download_model(model_id='THUDM/chatglm3-6b', commit_hash = "e46a14881eae613281abbd266ee918e93a56018f"):
    from huggingface_hub import snapshot_download
    from pathlib import Path

    local_model_folder_name = f"LLM_{model_id.replace('/', '_')}_model"
    local_code_folder_name = f"LLM_{model_id.replace('/', '_')}_code"

    s3_model_prefix = f"LLM-RAG/workshop/{local_model_folder_name}"  # folder where model checkpoint will go
    s3_code_prefix = f"LLM-RAG/workshop/{local_code_folder_name}"

    local_model_path = Path(local_model_folder_name)
    local_code_path = Path(local_code_folder_name)
    if(not local_model_path.exists()):
        local_model_path.mkdir(exist_ok=True)
        local_code_path.mkdir(exist_ok=True)
        snapshot_download(repo_id=model_id, revision=commit_hash, cache_dir=local_model_path)

        model_snapshot_path = list(local_model_path.glob("**/snapshots/*"))[0]

        print(f"s3_code_prefix: {s3_code_prefix}")
        print(f"model_snapshot_path: {model_snapshot_path}")

        os.system(f'aws s3 cp --recursive {model_snapshot_path} s3://{bucket}/{s3_model_prefix}')

    print(f'path {local_model_path} already exists')
    return local_code_folder_name, local_model_folder_name, s3_code_prefix, s3_model_prefix


def get_sm_role_arn():
    response = iam.get_role(RoleName=sagemaker_iam_role_name)
    role_arn = response['Role']['Arn']

    return role_arn

# a function to create iam role used for sagemaker service
def create_sm_role():
    response = iam.create_role(
        RoleName=sagemaker_iam_role_name,
        Description='Allows SageMaker endpoints to call other AWS services on your behalf',
        AssumeRolePolicyDocument=json.dumps({
            'Version': '2012-10-17', 
            'Statement': [{
                'Effect': 'Allow', 
                'Principal': {'Service': 'sagemaker.amazonaws.com'},
                'Action': 'sts:AssumeRole'
            }]
        })
    )
    iam.attach_role_policy(
        RoleName=sagemaker_iam_role_name,
        PolicyArn='arn:aws:iam::aws:policy/AdministratorAccess'
    )

    role_arn = response['Role']['Arn']
    return role_arn

import sagemaker
from sagemaker import image_uris

def create_llm_model_artifact(local_code_folder_name, s3_model_prefix, s3_code_prefix):
    with open(f'{local_code_folder_name}/model.py', 'w') as f:
        f.write("""
from djl_python import Input, Output
import torch
import logging
import math
import os

from transformers import pipeline, AutoModel, AutoTokenizer

def load_model(properties):
    tensor_parallel = properties["tensor_parallel_degree"]
    model_location = properties['model_dir']
    if "model_id" in properties:
        model_location = properties['model_id']
    logging.info(f"Loading model in {model_location}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_location, trust_remote_code=True)
   
    model = AutoModel.from_pretrained(model_location, trust_remote_code=True).half().cuda()
    
    # model.requires_grad_(False)
    # model.eval()
    
    return model, tokenizer


model = None
tokenizer = None
generator = None


def handle(inputs: Input):
    global model, tokenizer
    if not model:
        model, tokenizer = load_model(inputs.get_properties())

    if inputs.is_empty():
        return None
    data = inputs.get_as_json()
    
    input_sentences = data["inputs"]
    params = data["parameters"]
    history = data["history"]
    
    # chat(tokenizer, query: str, history: List[Tuple[str, str]] = None, 
    # max_length: int = 2048, num_beams=1, do_sample=True, top_p=0.7, 
    # temperature=0.95, logits_processor=None, **kwargs)
    response, history = model.chat(tokenizer, input_sentences, history=history, **params)
    
    result = {"outputs": response, "history" : history}
    return Output().add_as_json(result)
    """)

    print(f"option.s3url ==> s3://{bucket}/{s3_model_prefix}/")

    with open(f'{local_code_folder_name}/serving.properties', 'w') as f:
        f.write(f"""
engine=Python
option.tensor_parallel_degree=1
option.s3url = s3://{bucket}/{s3_model_prefix}/
        """)

    with open(f'{local_code_folder_name}/requirements.txt', 'w') as f:
        f.write(f"""transformers==4.28.1""")

    s3_code_artifact = f's3://{bucket}/{s3_code_prefix}/model.tar.gz'
    os.system(f"""
    rm model.tar.gz; \
    tar czvf model.tar.gz {local_code_folder_name}/ &&\
    aws s3 cp model.tar.gz {s3_code_artifact}
    """)

    print(f"S3 Code or Model tar ball uploaded to ---> {s3_code_artifact}")
    return s3_code_artifact

def create_llm_model(model_id, s3_code_artifact, iam_role_arn):
    from sagemaker.utils import name_from_base
    import boto3

    inference_image_uri = (
        f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.23.0-deepspeed0.9.5-cu118"
    )

    model_name = name_from_base(model_id.replace('/', '-'))
    print(model_name)
    print(f"Image going to be used is ----> {inference_image_uri}")

    create_model_response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=iam_role_arn,
        PrimaryContainer={
            "Image": inference_image_uri,
            "ModelDataUrl": s3_code_artifact
        },
    )
    model_arn = create_model_response["ModelArn"]

    print(f"Created Model: {model_arn}")

    time.sleep(1) # sleep before return
    return model_arn, model_name

def create_endpoint(model_name, iam_role_arn):
    endpoint_config_name = f"{model_name}-config"
    endpoint_name = f"{model_name}-endpoint"

    #Note: ml.g4dn.2xlarge 也可以选择
    endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "variant1",
                "ModelName": model_name,
                "InstanceType": "ml.g4dn.2xlarge",
                "InitialInstanceCount": 1,
                # "VolumeSizeInGB" : 400,
                # "ModelDataDownloadTimeoutInSeconds": 2400,
                "ContainerStartupHealthCheckTimeoutInSeconds": 15*60,
            },
        ],
    )

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=f"{endpoint_name}", EndpointConfigName=endpoint_config_name
    )
    print(f"Created Endpoint: {create_endpoint_response['EndpointArn']}")

    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    print("Status: " + status)

    with open('chatglm-sagemaker-endpoint.txt', 'w') as f:
        print('saving endpoint name to chatglm-sagemaker-endpoint.txt')
        f.write(endpoint_name)

    while status == "Creating":
        time.sleep(10)
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print("Status: " + status)

    print("Arn: " + resp["EndpointArn"])
    print("Status: " + status)

    return endpoint_name

def test_llm_model(endpoint_name):
    parameters = {
        "max_length": 2048,
        "temperature": 0.01,
        "num_beams": 1, # >1可能会报错，"probability tensor contains either `inf`, `nan` or element < 0"； 即使remove_invalid_values=True也不能解决
        "do_sample": False,
        "top_p": 0.7,
        "logits_processor" : None,
        # "remove_invalid_values" : True
    }

    prompts1 = """AWS Clean Rooms 的FAQ文档有提到 Q: 是否发起者和数据贡献者都会被收费？A: 是单方收费，只有查询的接收方会收费。
请问AWS Clean Rooms是多方都会收费吗？
    """
    response_model = smr_client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=json.dumps(
                {
                    "inputs": prompts1,
                    "parameters": parameters,
                    "history" : []
                }
                ),
                ContentType="application/json",
            )

    resp = response_model['Body'].read().decode('utf8')
    print(resp)


if __name__ == "__main__":
    try:
        iam_role_arn = create_sm_role()
    except:
        iam_role_arn = get_sm_role_arn()
        pass

    model_id = 'THUDM/chatglm3-6b'
    local_code_folder_name, local_model_folder_name, s3_code_prefix, s3_model_prefix = \
        download_model(model_id, "e46a14881eae613281abbd266ee918e93a56018f")
    s3_code_artifact = create_llm_model_artifact(local_code_folder_name, s3_model_prefix, s3_code_prefix)
    
    model_arn, model_name = create_llm_model(model_id, s3_code_artifact, iam_role_arn)
    endpoint_name = create_endpoint(model_name, iam_role_arn)
    test_llm_model(endpoint_name)