"""
runprobContextTask
"""

from adaptive_procedure import adaptiveThreshold
from make_config import ThresholdConfig
import numpy as np
from psychopy import event
import os
from twilio.rest import TwilioRestClient
from utils import get_trackers

# ****************************************************************************
# Helper Function
# ****************************************************************************
def send_message(msg):
    accountSid = 'AC0055c137ee1b1c3896f6c47389e487dc'
    twilioClient = TwilioRestClient(accountSid, authToken)
    twilio_info = open('../../twilio_info.txt','r')
    authToken = twilio_info.readline()
    myTwilioNumber = twilio_info.readline()
    destCellPhone = twilio_info.readline() 
    myMessage = twilioClient.messages.create(body = msg, from_=myTwilioNumber, to=destCellPhone)
        
        
# ****************************************************************************
# set-up variables
# ****************************************************************************

verbose=True
message_on = False
fullscr= True
subdata=[]
motion_on = True
color_on = False
home = os.getenv('HOME') 
save_dir = '../Data' 
motionname = 'adaptive_motion'
colorname = 'adaptive_color'
# set up task variables
subject_code = 'IE2'
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
motion_trackers,color_trackers = get_trackers(subject_code)

if motion_on:
    motion_config = ThresholdConfig(taskname=motionname, subjid=subject_code, 
                                        stim_repetitions=stim_repetitions,
                                        ts='motion',exp_len=exp_len)
    motion_config_file = motion_config.get_config()
    motion_task=adaptiveThreshold(motion_config_file,subject_code, 
                                  save_dir=save_dir, fullscreen=fullscr,
                                  trackers=motion_trackers)

if color_on:
    color_config = ThresholdConfig(taskname=colorname, subjid=subject_code, 
                                        stim_repetitions=stim_repetitions, 
                                        ts='color',exp_len=exp_len)
    color_config_file = color_config.get_config()  
    color_task=adaptiveThreshold(color_config_file,subject_code, 
                                  save_dir=save_dir, fullscreen=fullscr,
                                  trackers=color_trackers)




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
    
    #************************************
    # Send text about train performance
    #************************************
    if message_on == True:   
        send_message('Motion done')
        


if color_on:
    # prepare to start
    color_task.setupWindow()
    color_task.defineStims()
    color_task.presentTextToWindow("""Color""")
    resp,time=color_task.waitForKeypress(color_task.trigger_key)
    color_task.checkRespForQuitKey(resp)
    event.clearEvents()

    pause_trials = np.round(np.linspace(0,color_task.exp_len,n_pauses+2))[1:-1]
    color_task.run_task(pause_trials=pause_trials)    
    
    #************************************
    # Send text about train performance
    #************************************
    if message_on == True:   
        send_message('color done')


