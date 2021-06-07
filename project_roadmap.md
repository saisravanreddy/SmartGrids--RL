To do list:
1.Know how are sockets are initialized and how can we initialize a socket using custom port 
2.How can we debug multiple processes of the same python file in pycharm
3.pass different arguments to different debuggers informing which worker is it simulating
4.copy the code from replay method of the agent to the DQNLearner class
5.Use the prioritized replay buffer given in the distRL project
6.copy the environment from dqn_PER_trainer.py file to the environment_step method of DQNWorker class
7.copy the agent code to the remaining methods of the DQNWorker class 


1.Use pytorch itself(convert tensorflow code to pytorch code)

### Roadmap:
//update apex_dqn/run_apex_dqn.py
update common/abstract/learner.py
update config.yml
##### 3.update convDuelingDQN class to pricingDQN and adlDQN

1.Now configure the hyperparameters using config file

Ray multithreading is working as required, debug by disabling ray 

Issues:
Why are the losses different from the double DQN experiment graphs
why only 4 learning steps for every 1000 actions placed into buffer
learning actions are slow may be because of for loops 

learner is the bottleneck run worker for some time and then run buffer and learner
step by step

make all the performance improvements to the constant pricing  agent
and then change it to the dynamic pricing agent


may be get envpytorch to use gpu for faster learning








