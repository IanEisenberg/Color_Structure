"""
runprobContextTask
"""

import webbrowser
import numpy as np
from psychopy import event
from Exp_Design.make_config import ProbContextConfig
from Exp_Design.prob_context_task import probContextTask
from Exp_Design.utils import get_response_curve

# ****************************************************************************
# set-up variables
# ****************************************************************************
subjid = input('Enter the subject ID: ')

save_dir = '../Data' 
cuename = 'cued_dot_task'
cue_type = 'deterministic'
# set up task variables
stim_repetitions = 5
recursive_p = .9
# window variables
win_kwargs = {'fullscr': False,
              'screen': 1,
              'size': [1920, 1200]}
# counterbalance ts_order (which ts is associated with top of screen)
ts_order = ['motion','orientation']
np.random.shuffle(ts_order)



# ****************************************************************************
# set up config files
# ****************************************************************************
# load motion_difficulties and ori_difficulties from adaptive tasks     
# cued task 
responseCurves = get_response_curve(subjid)
cue_config = ProbContextConfig(taskname=cuename, 
                               subjid=subjid, 
                               stim_repetitions=stim_repetitions, 
                               ts_order=ts_order, 
                               rp=recursive_p,
                               speed_difficulties=[.7, .85],
                               ori_difficulties=[.7, .85],
                               responseCurves=responseCurves)
cue_config_file = cue_config.get_config(setup_args={'displayFB': False})

    

# setup tasks
cued_task=probContextTask(cue_config_file,
                          subjid, 
                          save_dir=save_dir,
                          cue_type=cue_type,
                          win_kwargs=win_kwargs)


# ****************************************************************************
# Start cueing
# ****************************************************************************

intro_text = \
    """
    In this phase of the experiment you will be cued
    whether to pay attention to motion or orientation
    before each trial.
    
    Please wait for the experimenter.
    """

cued_task.run_task(intro_text=intro_text)    
          
#************************************
# Determine payment
#************************************
points,trials = cued_task.getPoints()
performance = (float(points)/trials-.25)/.75
pay_bonus = round(performance*5)
print('Participant ' + subjid + ' won ' + str(points) + ' points out of ' + str(trials) + ' trials. Bonus: $' + str(pay_bonus))

#open post-task questionnaire
webbrowser.open_new('https://stanforduniversity.qualtrics.com/SE/?SID=SV_9KzEWE7l4xuORIF')






