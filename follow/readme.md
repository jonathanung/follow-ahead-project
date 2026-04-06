# Human Frontal Following for Mobile Robots

## Overview
This repository contains the implementation of a robot navigation system designed to follow a person in front. The system uses Monte Carlo Tree Search (MCTS) for decision-making and operates at a frequency of 5 Hz. 

## Requirments
The system needs ROS 1.

2D position of the robot and human: It can be provided by a Vicon system, camera, odometry, etc. The Vicon Mocap system is used in the provided code.

Costmap: Shows the locations of the occupied and free spaces around the robot.

## Key Modules

### Tree Expansion:

At each time step, the system expands a tree with two layers: robot nodes and human nodes. 
Robot nodes represent possible actions or states of the robot.
Human nodes represent potential states or movements of the person being followed.
The RL model evaluates each node to estimate its value.
The Human Trajcetory Prediction model assigns probabilitis to human nodes.

The expansion process is done by MCTS at a frequency of 5 Hz.
After expanding the tree, the child node with the highest visit count and expansion is chosen as the optimal action for the robot.
A new tree is expanded based on the updated state and inputs.


### Reinforcement Learning (RL) Model:

Evaluates the quality of tree nodes during the tree expansion.

### Human Trajcetory Prediction Model:

An LSTM based model assigns probabilities to human nodes based on history of human's postions.




## Usage

Run the main script to start the robot-following system:

Ensure that the vicion data and costmap inputs are provided.
