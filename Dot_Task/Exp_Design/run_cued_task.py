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
from twilio.rest import TwilioRestClient
from utils import get_difficulties

# ****************************************************************************
# Helper Function
# ****************************************************************************
def send_message(msg):
    accountSid = 'AC0055c137ee1b1c3896f6c47389e487dc'
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
home = os.getenv('HOME') 
save_dir = '../Data' 
cuename = 'cued_dot_task'
cue_type = 'deterministic'
n_pauses=3

"""
# set things up for practice, cueing and tests
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

# cued task 
cue_config = ProbContextConfig(taskname = cuename, 
                                 subjid = subject_code, 
                                 stim_repetitions = stim_repetitions, 
                                 ts_order = ts_order, rp = recursive_p,
                                 motion_difficulties = motion_difficulties,
                                 color_difficulties = color_difficulties)
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
# Send text about cue performance
#************************************
if message_on == True:   
    send_message('cueing done')
    

        
       
#************************************
# Determine payment
#************************************
points,trials = test.getPoints()
performance = (float(points)/trials-.25)/.75
pay_bonus = round(performance*5)
print('Participant ' + subject_code + ' won ' + str(points) + ' points out of ' + str(trials) + ' trials. Bonus: $' + str(pay_bonus))

#open post-task questionnaire
webbrowser.open_new('https://stanforduniversity.qualtrics.com/SE/?SID=SV_9KzEWE7l4xuORIF')






