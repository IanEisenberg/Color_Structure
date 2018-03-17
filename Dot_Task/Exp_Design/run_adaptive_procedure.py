"""
run_adaptive_procedure.py
--------------------------
Controls the sequence of the adaptive (calibration)
phase of the experiment.

Creats config file for subject and run the task.
-------------------------

@author: ian

"""
import random as r
from Dot_Task.Exp_Design.adaptive_procedure import adaptiveThreshold
from Dot_Task.Exp_Design.utils import get_trackers
from Dot_Task.Exp_Design.make_config import ThresholdConfig

        
# ****************************************************************************
# set-up variables
# ****************************************************************************
#get subject id
print('Enter the subject ID')
subjid = raw_input('subject id: ')

motion_on = True
orientation_on = True 
practice_on = False
eyetracker_on = False
one_difficulty=True
save_dir = '../Data' 
# set up task variables
stim_repetitions = 4
exp_len = None
# window variables
win_kwargs = {'fullscr': True,
              'allowGUI': True,
              'screen': 1,
              'size': [1920*.8, 1200*.8]}

# counterbalance ts_order (which ts is associated with top of screen)
first_task = 'motion' if r.random() > .5 else 'orientation'
first_task = 'orientation'

def setup_task(trackers, dim='motion', 
               speed_difficulties=None, ori_difficulties=None):
    config = ThresholdConfig(subjid=subjid, 
                             taskname='adaptive_%s' % dim,
                             stim_repetitions=stim_repetitions,
                             ts=dim,
                             exp_len=exp_len,
                             one_difficulty=one_difficulty)
    if speed_difficulties:
        config.speed_difficulties = speed_difficulties
    if ori_difficulties:
        config.ori_difficulties = ori_difficulties
    config_file = config.get_config()  
    task=adaptiveThreshold(config_file,
                           subjid,
                           save_dir=save_dir, 
                           win_kwargs=win_kwargs,
                           trackers=trackers[dim])
    return task

# ****************************************************************************
# ************** RUN TASK ****************************************************
# ****************************************************************************
trackers = get_trackers(subjid)

if first_task == 'motion':
    curr_task = setup_task(trackers, 'motion')
else:
    curr_task = setup_task(trackers, 'orientation')

done = False
while not done:
    done = curr_task.run_task(practice=practice_on,
                              eyetracker=eyetracker_on)
    # update trackers and swap task
    if curr_task.ts == 'motion':
        trackers['motion'] = curr_task.trackers
        curr_task = setup_task(trackers, 'orientation', 
                               speed_difficulties=curr_task.speed_difficulties)
    else:
        trackers['orientation'] = curr_task.trackers
        curr_task = setup_task(trackers, 'motion',
                               ori_difficulties=curr_task.ori_difficulties)

