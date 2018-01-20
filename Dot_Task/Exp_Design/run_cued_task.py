"""
runprobContextTask
"""

import webbrowser
from make_config import ProbContextConfig
import glob
import numpy as np
import os
from prob_context_task import probContextTask
from psychopy import event
from utils import get_difficulties

# ****************************************************************************
# set-up variables
# ****************************************************************************

verbose=True
fullscr= False
subdata=[]
save_dir = '../Data' 
cuename = 'cued_dot_task'
cue_type = 'deterministic'
n_pauses=3

subject_code = 'test'
# set up task variables
stim_repetitions = 5
recursive_p = .9

# counterbalance ts_order (which ts is associated with top of screen)
ts_order = ['motion','color']
try:
    if int(subject_code)%2 == 1:
        ts_order = ['color','motion']
except ValueError:
    pass


# ****************************************************************************
# set up config files
# ****************************************************************************
# load motion_difficulties and color_difficulties from adaptive tasks
motion_difficulties, ori_difficulties = get_difficulties(subject_code)

# cued task 
cue_config = ProbContextConfig(taskname = cuename, 
                                 subjid = subject_code, 
                                 stim_repetitions = stim_repetitions, 
                                 ts_order = ts_order, rp = recursive_p,
                                 motion_difficulties = motion_difficulties,
                                 color_difficulties = ori_difficulties)
cue_config_file = cue_config.get_config()

    

# setup tasks
cued_task=probContextTask(cue_config_file,subject_code, save_dir=save_dir, 
                      fullscreen = fullscr, cue_type=cue_type)


# ****************************************************************************
# ************** RUN TASK ****************************************************
# ****************************************************************************

# ****************************************************************************
# Start cueing
# ****************************************************************************

# prepare to start
cued_task.setupWindow()
cued_task.defineStims()
cued_task.presentTextToWindow(
    """
    We will now start the cued phase of the experiment.
    
    
    Please wait for the experimenter.
    """)
resp,time=cued_task.waitForKeypress(cued_task.trigger_key)
cued_task.checkRespForQuitKey(resp)
event.clearEvents()

pause_trials = np.round(np.linspace(0,cued_task.exp_len,n_pauses+2))[1:-1]
cued_task.run_task(pause_trials=pause_trials)    
        
       
#************************************
# Determine payment
#************************************
points,trials = cued_task.getPoints()
performance = (float(points)/trials-.25)/.75
pay_bonus = round(performance*5)
print('Participant ' + subject_code + ' won ' + str(points) + ' points out of ' + str(trials) + ' trials. Bonus: $' + str(pay_bonus))

#open post-task questionnaire
webbrowser.open_new('https://stanforduniversity.qualtrics.com/SE/?SID=SV_9KzEWE7l4xuORIF')






