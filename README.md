# Learning Deformable Object Model with GNS
Codes for learning deformable object model with GNS

## Install
- create conda environment: ```conda create -n deformable_gns python=3.7 pip```
- run ```pip install pybullet``` for pybullet
- Install requirements files according to https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate (requires Python3.7-): ```pip install -r learning_to_simulate/requirements.txt```
- Downgrade Protobuf to avoid potential problems of tensorflow: ```pip install --upgrade "protobuf<=3.20.1"```
- Install sklearn: ```pip install -U scikit-learn```
- Install IPOPT: ```sudo apt-get install gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev```
- Install IPOPT python wrapper: ```conda install -c conda-forge cyipopt``` (required Python 3.6+)
- Install opencv: ```conda install -c conda-forge opencv```

## Usage
- You can use ```chmod +x run.sh``` & ```run.sh``` for collecting the data and train the network
- For details, run ``` python collect_rope_data_2d.py``` for collecting rope data in tfrecord format.
- You can use ```python learning_to_simulate/show_data_message.py``` to test whether the data is recorded correctly.
- run ```python learning_to_simulate/evaluate.py``` for generating rollouts
- run ```python learning_to_simulate/render_rope.py``` for the results
- run ```python planning.py``` for planning example (a pre-trained model was put in the learning_to_simulate/models)
- for speeding up the planning, you can play with the optimizaiton options ```tol``` and ```max_iter```

## Troubleshooting
Contact changhaowang@berkeley.edu