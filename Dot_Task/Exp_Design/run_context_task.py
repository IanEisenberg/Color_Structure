"""
runprobContextTask
"""

import webbrowser
from make_config import ProbContextConfig
import glob
import numpy as np
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
train_on = True
test_on = False
save_dir = '../Data' 
trainname = 'dot_task'
cue_type = 'probabilistic'
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
     (-60, 'easy'): 15,
     (-60, 'hard'): 5,
     (30, 'easy'): 15,
     (30, 'hard'): 5}   
    
# train 
if train_on:
    train_config = ProbContextConfig(taskname=trainname, 
                                     subjid=subject_code, 
                                     stim_repetitions=stim_repetitions, 
                                     ts_order=ts_order, 
                                     rp=recursive_p,
                                     motion_difficulties=motion_difficulties,
                                     ori_difficulties=ori_difficulties)
    train_config_file = train_config.get_config()
else:
    train_config_file = glob.glob('../Config_Files/*Context_' +subject_code +'*yaml')[-1]
    
test_config = ProbContextConfig(taskname=train_config.taskname+'_test',
                                subjid=subject_code,
                                motion_difficulties=motion_difficulties, 
                                ori_difficulties=ori_difficulties)
test_config.load_config_settings(train_config_file, 
                                 stim_repetitions=stim_repetitions)
test_config.setup_trial_list(displayFB = False)
test_config_file = test_config.get_config()

# setup tasks
train=probContextTask(train_config_file,
                      subject_code, 
                      save_dir=save_dir, 
                      fullscreen = fullscr, 
                      cue_type=cue_type)
test=probContextTask(test_config_file,
                     subject_code, 
                     save_dir=save_dir, 
                     fullscreen = fullscr, 
                     cue_type=cue_type)


# ****************************************************************************
# ************** RUN TASK ****************************************************
# ****************************************************************************

# ****************************************************************************
# Start training
# ****************************************************************************

if train_on:

    intro_text = \
        """
        We will now start the training phase of the experiment.
        
        Remember, following this training phase will be a test phase with no
        feedback (you won't see points). Use this training to learn when
        you have to respond to the identity or color of the shape without
        needing to use the points.
        
        There will be one break half way through. As soon
        as you press '5' the experiment will start so get ready!
        
        Please wait for the experimenter.
        """

    train.run_task(intro_text=intro_text)    
    
# ****************************************************************************
# Start test
# ****************************************************************************
        
if test_on:
    intro_text = \
    """
        In this next part the feedback will be invisible. You
        are still earning points, though, and these points are
        used to determine your bonus.
        
        Do your best to respond to the shapes as you learned to
        in the last section.
        
        Please wait for the experimenter.
        """
                        
    test.run_task(intro_text=intro_text)    
        
#************************************
# Determine payment
#************************************
points,trials = test.getPoints()
performance = (float(points)/trials-.25)/.75
pay_bonus = round(performance*5)
print('Participant ' + subject_code + ' won ' + str(points) + ' points out of ' + str(trials) + ' trials. Bonus: $' + str(pay_bonus))

#open post-task questionnaire
webbrowser.open_new('https://stanforduniversity.qualtrics.com/SE/?SID=SV_9KzEWE7l4xuORIF')






