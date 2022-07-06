# MCTS for Game Playing 🕹

This project was part of the Artificial Intelligence Programming (IT3105) course at NTNU spring 2022. The aim of this project was to implement an Monte Carlo Tree Search (MCTS) system and combine it with Reinforcement Learning (RL) and Deep Learning (DL) to play the board game Hex.

## Monte Carlo Tree Search

In this project an on-policy MCTS approach (i.e., the target policy and behavior policy are the same) was employed,
and that policy was implemented as a neural network. MCTS then provided training cases for that policy network,
in the same spirit as Google DeepMind’s use of MCTS to train deep networks for playing Go, chess and shogi. That
same network, in fulfilling its role as the behavior policy, guided rollout simulations in MCTS.
