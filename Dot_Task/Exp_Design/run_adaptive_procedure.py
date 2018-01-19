"""
runprobContextTask
"""

from adaptive_procedure import adaptiveThreshold
from make_config import ThresholdConfig
import numpy as np
from psychopy import event
import os
from utils import get_trackers

        
# ****************************************************************************
# set-up variables
# ****************************************************************************

verbose=True
message_on = False
fullscr= False
subdata=[]
motion_on = False
orientation_on = True
home = os.getenv('HOME') 
save_dir = '../Data' 
motionname = 'adaptive_motion'
orientationname = 'adaptive_orientation'
# set up task variables
subject_code = 'IE22'
stim_repetitions = 4
exp_len = None
n_pauses=1

"""
# set things up for practice, training and tests
try:
    f = open('IDs.txt','r')
    lines = f.readlines()
    f.close()
    try:
        last_id = lines[-1][:-1]
        subject_code = raw_input('Last subject: "%s". Input new subject code: ' % last_id);
    except IndexError:
        subject_code = raw_input('Input first subject code: ');
except IOError:
    subject_code = raw_input('Input first subject code: ');
f = open('IDs.txt', 'a')
f.write(subject_code + '\n')
f.close()
"""


# counterbalance ts_order (which ts is associated with top of screen)
ts_order = ['motion','orientation']
try:
    if int(subject_code)%2 == 1:
        ts_order = ['orientation','motion']
except ValueError:
    pass

# ****************************************************************************
# set up config files
# ****************************************************************************
# load motion_difficulties and orientation_difficulties from adaptive tasks
motion_trackers,orientation_trackers = get_trackers(subject_code)

if motion_on:
    motion_config = ThresholdConfig(taskname=motionname, subjid=subject_code, 
                                        stim_repetitions=stim_repetitions,
                                        ts='motion',exp_len=exp_len)
    motion_config_file = motion_config.get_config()
    motion_task=adaptiveThreshold(motion_config_file,subject_code, 
                                  save_dir=save_dir, fullscreen=fullscr,
                                  trackers=motion_trackers)

if orientation_on:
    orientation_config = ThresholdConfig(taskname=orientationname, subjid=subject_code, 
                                        stim_repetitions=stim_repetitions, 
                                        ts='orientation',exp_len=exp_len)
    orientation_config_file = orientation_config.get_config()  
    orientation_task=adaptiveThreshold(orientation_config_file,subject_code, 
                                  save_dir=save_dir, fullscreen=fullscr,
                                  trackers=orientation_trackers)




# ****************************************************************************
# ************** RUN TASK ****************************************************
# ****************************************************************************

# ****************************************************************************
# Start training
# ****************************************************************************

if motion_on:
    # prepare to start
    motion_task.setupWindow()
    motion_task.defineStims()
    motion_task.presentTextToWindow("""Motion""")
    resp,time=motion_task.waitForKeypress(motion_task.trigger_key)
    motion_task.checkRespForQuitKey(resp)
    event.clearEvents()
    
    pause_trials = np.round(np.linspace(0,motion_task.exp_len,n_pauses+2))[1:-1]
    motion_task.run_task(pause_trials=pause_trials)    
    


if orientation_on:
    # prepare to start
    orientation_task.setupWindow()
    orientation_task.defineStims()
    orientation_task.presentTextToWindow("""Orientation""")
    resp,time=orientation_task.waitForKeypress(orientation_task.trigger_key)
    orientation_task.checkRespForQuitKey(resp)
    event.clearEvents()

    pause_trials = np.round(np.linspace(0,orientation_task.exp_len,n_pauses+2))[1:-1]
    orientation_task.run_task(pause_trials=pause_trials)    
