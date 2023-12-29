sagemaker_iam_role_name='sm-execution-role-private-rag-workshop'

def get_cfn_output(region='us-west-2', cfn_name='QAChatDeployStack'):
    import boto3

    cf_client = boto3.client('cloudformation', region_name=region)

    response = cf_client.describe_stacks(StackName=cfn_name)

    _outputs = response['Stacks'][0]['Outputs']

    outputs = {}
    for item in _outputs:
        outputs[item['OutputKey']] = item['OutputValue']

    return outputs


