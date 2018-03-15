"""
run_adaptive_procedure.py
--------------------------
Controls the sequence of the adaptive (calibration)
phase of the experiment.

Creats config file for subject and run the task.
-------------------------

@author: ian

"""

from Dot_Task.Exp_Design.adaptive_procedure import adaptiveThreshold
from Dot_Task.Exp_Design.utils import get_trackers
from Dot_Task.Exp_Design.make_config import ThresholdConfig

        
# ****************************************************************************
# set-up variables
# ****************************************************************************
#get subject id
print('Enter the subject ID')
subject_code = raw_input('subject id: ')

motion_on = True
orientation_on = True 
practice_on = False
eyetracker_on = False
one_difficulty=True
save_dir = '../Data' 
# set up task variables
stim_repetitions = 2
exp_len = None
# window variables
win_kwargs = {'fullscr': False,
              'allowGUI': True,
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
trackers = get_trackers(subject_code)

if motion_on:
    motion_config = ThresholdConfig(subjid=subject_code, 
                                    taskname='adaptive_motion',
                                    stim_repetitions=stim_repetitions,
                                    ts='motion',
                                    exp_len=exp_len,
                                    one_difficulty=one_difficulty)
    motion_config_file = motion_config.get_config()
    motion_task=adaptiveThreshold(motion_config_file,
                                  subject_code, 
                                  save_dir=save_dir, 
                                  win_kwargs=win_kwargs,
                                  trackers=trackers['motion'])

if orientation_on:
    orientation_config = ThresholdConfig(subjid=subject_code, 
                                         taskname='adaptive_orientation',
                                         stim_repetitions=stim_repetitions,
                                         ts='orientation',
                                         exp_len=exp_len,
                                         one_difficulty=one_difficulty)
    orientation_config_file = orientation_config.get_config()  
    orientation_task=adaptiveThreshold(orientation_config_file,
                                       subject_code,
                                       save_dir=save_dir, 
                                       win_kwargs=win_kwargs,
                                       trackers=trackers['orientation'])


# ****************************************************************************
# ************** RUN TASK ****************************************************
# ****************************************************************************


if ts_order == ['motion', 'orientation']:
    if motion_on:
        motion_task.run_task(practice=practice_on,
                             eyetracker=eyetracker_on)    
    if orientation_on:
        orientation_task.run_task(practice=practice_on,
                                  eyetracker=eyetracker_on)    
else:
    if orientation_on:
        orientation_task.run_task(practice=practice_on,
                                  eyetracker=eyetracker_on) 
    if motion_on:
        motion_task.run_task(practice=practice_on,
                             eyetracker=eyetracker_on)
        
        

