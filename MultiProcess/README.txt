
To run the MCTS with a joint network (policy and value networks share encoder), run ./runMulti
To run the MCTS with disjoint networks, run ./runMultiDisjoint

Currently runMulti only uses the policy but can reuse the value network with only a couple changes to 
the policy_net.py file which defines how the forward pass of the network is run as well as how our loss is computed. 

