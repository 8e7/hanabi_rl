Python version: 3.11

Installation:
```
pip install -r requirements.txt
cd hanabi_learning_environment
pip install .
```

Basic Structure:
```
train.py -> Training script. Note that the number of subprocenvs must be 20, check line 131 for more detail.
eval.py -> Evaluate script for sb models.
eval_multi.py -> Evaluate script for sb model against sad agent.

training_agents.txt / testing_agents.txt -> list of agents where each line is the path to the weight file.
sad_code/ -> code for loading sad model, also has a modified implementation of r2d2 agent.
envs/ -> environments
- hanabi_env.py -> default hanabi training env (self play)
- hanabi_eval_env.py -> hanabi eval env (self play)
- hanabi_multi_train_env.py -> hanabi training env to play with sad / op agents.
- hanabi_multi_eval_env.py -> hanabi eval env to play with sad / op agents.
```
