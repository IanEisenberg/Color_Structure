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
from twilio.rest import Client
from utils import get_difficulties

# ****************************************************************************
# Helper Function
# ****************************************************************************
def send_message(msg):
    accountSid = 'AC0055c137ee1b1c3896f6c47389e487dc'
    twilioClient = Client(accountSid, authToken)
    twilio_info = open('../../twilio_info.txt','r')
    authToken = twilio_info.readline()
    twilioClient = TwilioRestClient(accountSid, authToken)
    myTwilioNumber = twilio_info.readline()
    destCellPhone = twilio_info.readline() 
    myMessage = twilioClient.messages.create(body = msg, from_=myTwilioNumber, to=destCellPhone)
        
        
# ****************************************************************************
# set-up variables
# ****************************************************************************

verbose=True
message_on = False
fullscr= False
subdata=[]
train_on = True
test_on = False
home = os.getenv('HOME') 
save_dir = '../Data' 
trainname = 'Dot_task'
cue_type = 'determinstic'
n_pauses=3

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
subject_code = 'IE'
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
motion_difficulties, color_difficulties = get_difficulties(subject_code)

# train 
if train_on:
    train_config = ProbContextConfig(taskname = trainname, 
                                     subjid = subject_code, 
                                     stim_repetitions = stim_repetitions, 
                                     ts_order = ts_order, rp = recursive_p,
                                     motion_difficulties = motion_difficulties,
                                     color_difficulties = color_difficulties)
    train_config_file = train_config.get_config()
else:
    train_config_file = glob.glob('../Config_Files/*Context_' +subject_code +'*yaml')[-1]
    
test_config = ProbContextConfig(motion_difficulties, color_difficulties)
test_config.load_config_settings(train_config_file, taskname=train_config.taskname+'_test', stim_repetitions = stim_repetitions)
test_config.setup_trial_list(displayFB = False)
test_config_file = test_config.get_config()

# setup tasks
train=probContextTask(train_config_file,subject_code, save_dir=save_dir, 
                      fullscreen = fullscr, cue_type=cue_type)
test=probContextTask(test_config_file,subject_code, save_dir=save_dir, 
                     fullscreen = fullscr, cue_type=cue_type)


# ****************************************************************************
# ************** RUN TASK ****************************************************
# ****************************************************************************

# ****************************************************************************
# Start training
# ****************************************************************************

if train_on:
    # prepare to start
    train.setupWindow()
    train.defineStims()
    train.presentTextToWindow(
        """
        We will now start the training phase of the experiment.
        
        Remember, following this training phase will be a test phase with no
        feedback (you won't see points). Use this training to learn when
        you have to respond to the identity or color of the shape without
        needing to use the points.
        
        There will be one break half way through. As soon
        as you press '5' the experiment will start so get ready!
        
        Please wait for the experimenter.
        """)
    resp,time=train.waitForKeypress(train.trigger_key)
    train.checkRespForQuitKey(resp)
    event.clearEvents()

    pause_trials = np.round(np.linspace(0,train.exp_len,n_pauses+2))[1:-1]
    train.run_task(pause_trials=pause_trials)    
    
    #************************************
    # Send text about train performance
    #************************************
    if message_on == True:   
        send_message('Training done')
        

        
# ****************************************************************************
# Start test
# ****************************************************************************
        
if test_on:
    # prepare to start
    test.setupWindow()
    test.defineStims()
    test.presentTextToWindow(
        """
        In this next part the feedback will be invisible. You
        are still earning points, though, and these points are
        used to determine your bonus.
        
        Do your best to respond to the shapes as you learned to
        in the last section.
        
        Please wait for the experimenter.
        """)
                        
    resp,time=test.waitForKeypress(test.trigger_key)
    test.checkRespForQuitKey(resp)
    event.clearEvents()
        
    pause_trials = np.round(np.linspace(0,test.exp_len,n_pauses+2))[1:-1]
    test.run_task(pause_trials=pause_trials)    
        
    #************************************
    # Send text about test performance
    #************************************
    if message_on == True:   
        send_message('Testing Done')
       
#************************************
# Determine payment
#************************************
points,trials = test.getPoints()
performance = (float(points)/trials-.25)/.75
pay_bonus = round(performance*5)
print('Participant ' + subject_code + ' won ' + str(points) + ' points out of ' + str(trials) + ' trials. Bonus: $' + str(pay_bonus))

#open post-task questionnaire
webbrowser.open_new('https://stanforduniversity.qualtrics.com/SE/?SID=SV_9KzEWE7l4xuORIF')






