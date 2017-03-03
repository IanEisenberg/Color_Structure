"""
runprobContextTask
"""

from psychopy import event
import webbrowser
from threshold_procedure import adaptiveThreshold
from make_config import ThresholdConfig
import glob
import os
from twilio.rest import TwilioRestClient

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
subject_code = 'test'
# set up task variables
stim_repetitions = 5

# counterbalance ts_order (which ts is associated with top of screen)
try:
    if int(subject_code)%2 == 0:
        ts_order = [0,1]
    else:
        ts_order = [1,0]
except ValueError:
    ts_order = [0,1]

# ****************************************************************************
# set up config files
# ****************************************************************************
# train 


if motion_on:
    motion_config = ThresholdConfig(taskname=motionname, subjid=subject_code, 
                                        stim_repetitions=stim_repetitions,
                                        ts='motion',)
    motion_config_file = motion_config.get_config()
    motion_task=adaptiveThreshold(motion_config_file,subject_code, 
                                  save_dir=save_dir, fullscreen = fullscr)

if color_on:
    color_config = ThresholdConfig(taskname=colorname, subjid=subject_code, 
                                        stim_repetitions=stim_repetitions, 
                                        ts='color')
    color_config_file = color_config.get_config()  
    color_task=adaptiveThreshold(color_config_file,subject_code, 
                                  save_dir=save_dir, fullscreen = fullscr)




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
    motion_task.presentTextToWindow(
        """
        Motion
        """)
    resp,time=motion_task.waitForKeypress(motion_task.trigger_key)
    motion_task.checkRespForQuitKey(resp)
    event.clearEvents()

    pause_trial = motion_task.stimulusInfo[len(motion_task.stimulusInfo)/2]
    motion_task.run_task(pause_trial=pause_trial)    
    
    #************************************
    # Send text about train performance
    #************************************
    if message_on == False:   
        send_message('Motion done')
        


if color_on:
    # prepare to start
    color_task.setupWindow()
    color_task.defineStims()
    color_task.presentTextToWindow(
        """
        Color
        """)
    resp,time=color_task.waitForKeypress(color_task.trigger_key)
    color_task.checkRespForQuitKey(resp)
    event.clearEvents()

    pause_trial = color_task.stimulusInfo[len(color_task.stimulusInfo)/2]
    color_task.run_task(pause_trial=pause_trial)    
    
    #************************************
    # Send text about train performance
    #************************************
    if message_on == False:   
        send_message('color done')


