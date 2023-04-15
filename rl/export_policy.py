# open model zip file and extract the model
import zipfile
import os
import sys
import gymnasium as gym
import robosuite_envs
import pointcloud_vision
from sb3_contrib import TQC
from sb3_contrib.tqc.policies import MultiInputPolicy

def save_policy(file):
    # to = os.path.dirname(file) + '/'
    task = os.path.basename(file).replace('.zip', '')
    print(task)

    # zip_ref = zipfile.ZipFile(file, 'r')
    # if 'policy.pth' in zip_ref.namelist():
    #     zip_ref.extract('policy.pth', to)
    #     # rename the extracted file
    #     os.rename(to + 'policy.pth', file.replace('.zip', '_policy.pth'))
    # zip_ref.close()

    env = gym.make(task)
    model = TQC.load(file, env=env)
    model.policy.save(file.replace('.zip', '_policy'))
    env.close()
    
if __name__ == '__main__':
    save_policy(sys.argv[1])