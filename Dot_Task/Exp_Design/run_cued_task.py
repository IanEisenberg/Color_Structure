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
print('Enter the subject ID')
subject_code = raw_input('subject id: ')

verbose=True
fullscr= False
subdata=[]
save_dir = '../Data' 
cuename = 'cued_dot_task'
cue_type = 'deterministic'
# set up task variables
stim_repetitions = 5
recursive_p = .9

# counterbalance ts_order (which ts is associated with top of screen)
ts_order = ['motion','orientation']
np.random.shuffle(ts_order)



# ****************************************************************************
# set up config files
# ****************************************************************************
# load motion_difficulties and ori_difficulties from adaptive tasks
motion_difficulties, ori_difficulties = get_difficulties(subject_code)
if motion_difficulties == {}:
    motion_difficulties = {
     ('in', 'easy'): 0.05,
     ('in', 'hard'): 0.01,
     ('out', 'easy'): 0.05,
     ('out', 'hard'): 0.01}
    
if ori_difficulties == {}:
     ori_difficulties = {
     (-60, 'easy'): 5,
     (-60, 'hard'): 15,
     (30, 'easy'): 5,
     (30, 'hard'): 15}   
     
# cued task 
cue_config = ProbContextConfig(taskname=cuename, 
                               subjid=subject_code, 
                               stim_repetitions=stim_repetitions, 
                               ts_order=ts_order, 
                               rp=recursive_p,
                               motion_difficulties=motion_difficulties,
                               ori_difficulties=ori_difficulties)
cue_config_file = cue_config.get_config()

    

# setup tasks
cued_task=probContextTask(cue_config_file,
                          subject_code, 
                          save_dir=save_dir, 
                          fullscreen=fullscr, 
                          cue_type=cue_type)


# ****************************************************************************
# ************** RUN TASK ****************************************************
# ****************************************************************************

# ****************************************************************************
# Start cueing
# ****************************************************************************

intro_text = \
    """
    We will now start the cued phase of the experiment.
    
    
    Please wait for the experimenter.
    """

cued_task.run_task(intro_text=intro_text)    
          
#************************************
# Determine payment
#************************************
points,trials = cued_task.getPoints()
performance = (float(points)/trials-.25)/.75
pay_bonus = round(performance*5)
print('Participant ' + subject_code + ' won ' + str(points) + ' points out of ' + str(trials) + ' trials. Bonus: $' + str(pay_bonus))

#open post-task questionnaire
webbrowser.open_new('https://stanforduniversity.qualtrics.com/SE/?SID=SV_9KzEWE7l4xuORIF')






