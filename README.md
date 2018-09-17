# simple_automl
A simple AutoML example with only one file. And adapt to all existing code.

## This project is 'Build in Progress', thanks for any contribution. 

## usage
### run mnist example
- let train.py to be your training script, then run executer.py to run experiment.
- executer.py will setup train.py for max_exp times, and work until stop_policy  in example_policy

### modify for your experiments
- Just implement a new policy like example_policy in executer.py, you can do anything to prepare your hyper parameters: Using LSTM to perdict hyper parameters, Random search, and etc.
- But, remenber to return hyper parameters as string to call your training script

### TODO
- [ ] Experiments restore
- [ ] Write all logs with tensorboard
- [ ] Multi-GPU scheduling
- [ ] Advanced tunning policy 
