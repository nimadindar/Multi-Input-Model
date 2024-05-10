import subprocess
import os
import numpy as np

import os

# The mode has to be either train or test
mode = 'train'

cwd = os.getcwd()
cwd = cwd.replace('\\','/')
# Path to your configuration files
results_folder = '/Results/cfgs'

listdir = os.listdir(cwd+results_folder)

for Testfile in listdir:
    # testId gets the number at the end of text configuration file name
    # e.g. test12.txt  ----> testId = 12
    testId = int(Testfile.replace('test', '').split('.')[0])
    command = ["python", os.path.join(cwd, "model.py"), f"--mode={mode}", 
                                                        f"--test_number={testId}"]
    subprocess.run(command)
    with open(cwd+'run_num_start.txt', 'w') as run_num_start_file:
        np.savetxt(run_num_start_file, [testId], fmt='%s')