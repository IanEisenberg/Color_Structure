Probabilistic Context Task
======

Psychopy task that presents four stimuli varying in color (red/blue) and shape (circle/square),
requiring the subjects to select one of four actions. During a training phase, subjects
get deterministic feedback indicating whether they performed the correct action. A test
phase follows where the feedback is removed.

Correct responses are determined by the currently operating task-set.
Task-sets alternate probabilistically and require subjects to either respond based on
the shape or color of the stimulus. Each task-set is probabilistically cued by the
stimuli's vertical position on the screen. Each task-set is associated with one of
two gaussian distributions determining the stimuli's position, either centered above or
below the screen's midline. Thus when task-set A is operating, the stimuli may be presented
on the top half of the screen more often.

Thus the task requires subjects to infer a latent state (the currently operating task-set)
based on a probabilistic cue (the stimuli's position) in order to determine the appropriate
action policy to take based on the stimuli.

-run_struct_task: Initialization file. 
-prob_context_task: Defines a probabilistic context task. Options determine whether the task
	is training (and thus has FB) or test (and thus doesn't)
-test_bot: a simple optimal observer decision maker
-make_config: defines a trial set. 