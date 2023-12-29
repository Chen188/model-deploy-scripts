import sys, os
import time

if len(sys.argv) !=2:
    print("Usage: python3 setup.py <region-name>")
    exit(1)

region=sys.argv[1]

files_to_watch = []
if not os.path.exists('bge-sagemaker-endpoint.txt'):
    files_to_watch.append('bge-sagemaker-endpoint.txt')
    os.system(f"python deploy-bge.py {region} &")
    print('deploy bge model...')
else:
    print('skip bge cause bge-sagemaker-endpoint.txt alread exist')

if not os.path.exists('chatglm-sagemaker-endpoint.txt'):
    files_to_watch.append('chatglm-sagemaker-endpoint.txt')
    os.system(f"python deploy-chatglm.py {region} &")
    print('deploy chatglm model...')
else:
    print('skip chatglm cause chatglm-sagemaker-endpoint.txt alread exist')

for i in range(60):
    not_ready_cnt = 0

    for file in files_to_watch:
        if not os.path.exists(file):
            not_ready_cnt += 1

    if not_ready_cnt == 0:
        print('all endpoint config found.')
        break
    else:
        print('still waiting for deployment...')
        time.sleep(10)


print('update lambda env...')
os.system(f'python update-lambda-env.py {region}')