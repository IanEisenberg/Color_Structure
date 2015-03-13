Temporal Structure Task
======

Psychopy task that presents temporally alternating task-sets. These task-sets require 
that subject's learn to correctly pair one of four actions with one of two stimuli. The
correct pairings change over time (alternating in a structured fashion, i.e. every 10 
trials), thus the goal of the learner is to learn the two task-sets and the overall
temporal structure.

-run_struct_task: Initialization file. 
-temp_struct_task: Defines a Temporal Structure Training session
-temp_struct_test_noFB: Defines a test phase identical to the training phase with no FB
-test_bot: a Q-learning bot that can perform the task in a flat manner 
(will not learn temporal structure)
-make_config: defines a trial set. 