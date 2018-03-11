"""
run_adaptive_procedure.py
--------------------------
Controls the sequence of the adaptive (calibration)
phase of the experiment.

Creats config file for subject and run the task.
-------------------------

@author: ian

"""

from adaptive_procedure import adaptiveThreshold
from make_config import ThresholdConfig
import os
from utils import get_trackers

        
# ****************************************************************************
# set-up variables
# ****************************************************************************
#get subject id
print('Enter the subject ID')
subject_code = raw_input('subject id: ')

verbose=True
message_on = False
subdata=[]
motion_on = True
orientation_on = True #True
practice_on = False
home = os.getenv('HOME') 
save_dir = '../Data' 
motionname = 'adaptive_motion'
orientationname = 'adaptive_orientation'
# set up task variables
stim_repetitions = 2
exp_len = None
# window variables
win_kwargs = {'fullscr': False,
              'screen': 1,
              'size': [1920*.8, 1200*.8]}

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
    motion_config = ThresholdConfig(subjid=subject_code, 
                                    taskname='adaptive_motion',
                                    stim_repetitions=stim_repetitions,
                                    ts='motion',
                                    exp_len=exp_len)
    motion_config_file = motion_config.get_config()
    motion_task=adaptiveThreshold(motion_config_file,
                                  subject_code, 
                                  save_dir=save_dir, 
                                  win_kwargs=win_kwargs,
                                  trackers=motion_trackers)

if orientation_on:
    orientation_config = ThresholdConfig(subjid=subject_code, 
                                         taskname='adaptive_orientation',
                                         stim_repetitions=stim_repetitions,
                                         ts='orientation',
                                         exp_len=exp_len)
    orientation_config_file = orientation_config.get_config()  
    orientation_task=adaptiveThreshold(orientation_config_file,
                                       subject_code,
                                       save_dir=save_dir, 
                                       win_kwargs=win_kwargs,
                                       trackers=orientation_trackers)

# ****************************************************************************
# ************** INTRODUCTION AND PRACTICE ***********************************
# ****************************************************************************



# ****************************************************************************
# ************** RUN TASK ****************************************************
# ****************************************************************************

# ****************************************************************************
# Start training
# ****************************************************************************
if ts_order == ['motion', 'orientation']:
    if motion_on:
        motion_task.run_task(practice=practice_on)    
    if orientation_on:
        orientation_task.run_task(practice=practice_on)    
else:
    if orientation_on:
        orientation_task.run_task(practice=practice_on) 
    if motion_on:
        motion_task.run_task(practice=practice_on)
        
        

