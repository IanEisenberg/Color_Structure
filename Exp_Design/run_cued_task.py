"""
runprobContextTask
"""

from Exp_Design.make_config import ProbContextConfig, split_config
from Exp_Design.prob_context_task import probContextTask
from Exp_Design.utils import get_response_curves


# ****************************************************************************
# set-up variables
# ****************************************************************************
subjid = input('Enter the subject ID: ')
fmri = True if input('fmri scan? y/n: ') == 'y' else False

save_dir = '../Data' 
cuename = 'cued_dot_task'
cue_type = 'deterministic'
# set up task variables
stim_repetitions = 1
recursive_p = .5
# window variables
win_kwargs = {'fullscr': False,
              'screen': 1,
              'size': [1920, 1200]}
action_keys = ['down','up','left','right']
fmri_trigger=None
# set up for fmri
if fmri == True:
    action_keys = ['e', 'b','r','y']
    fmri_trigger = 'quoteleft'



# ****************************************************************************
# set up config files
# ****************************************************************************
# load motion_difficulties and ori_difficulties from adaptive tasks     
# cued task 
responseCurves = get_response_curves(subjid)
cue_config = ProbContextConfig(taskname=cuename, 
                               subjid=subjid, 
                               action_keys=action_keys,
                               stim_repetitions=stim_repetitions, 
                               rp=recursive_p,
                               speed_difficulties=[.75],
                               ori_difficulties=[.75],
                               responseCurves=responseCurves)
complete_config = cue_config.get_config(setup_args={'displayFB': False, 'counterbalance_task': True}, 
                                        save=False)
# split into subconfigs        
last_trial = complete_config[-1]
total_length = last_trial['onset']/60
num_trials = last_trial['trial_count']
trials_per_run = int(7/total_length*num_trials)
config_files = split_config(cue_config, trials_per_run=trials_per_run)

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

for config_file in config_files:
    cued_task=probContextTask(config_file,
                          subjid, 
                          save_dir=save_dir,
                          cue_type=cue_type,
                          fmri_trigger=fmri_trigger,
                          win_kwargs=win_kwargs)
    if fmri == True:
        cued_task.run_task(intro_text=None)   
    else:
        cued_task.run_task(intro_text=intro_text)   
    exp_continue = True if input('Coninue? y/n: ') == 'y' else False
    if not exp_continue:
        break

              



