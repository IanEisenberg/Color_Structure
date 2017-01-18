"""
runprobContextTask
"""

from psychopy import core, event
import webbrowser
from prob_context_task import probContextTask
from make_config import ConfigList
from test_bot import test_bot
import glob
import os
from twilio.rest import TwilioRestClient

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
fullscr= True
subdata=[]
practice_on = True
train_on = True
test_on = True
bot_on = False
bot_mode = "ignore_base" #other for optimal
home = os.getenv('HOME') 
save_dir = '../Data' 
trainname = 'Prob_Context'

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

# set up task variables
train_mins = 45 #train_length in minutes
test_mins = 30 #test_length in minutes
avg_test_trial_len = 2.25 #in seconds
avg_train_trial_len = avg_test_trial_len + 1 #factor in FB
# Find the minimum even number of blocks to last the length of train/test
train_len = int(round(train_mins*60/avg_train_trial_len/4)*4)
test_len = int(round(test_mins*60/avg_test_trial_len/4)*4)
recursive_p = .9

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

# practice
practice_config_file = '../Config_Files/Prob_Context_Practice_config.yaml'
if not os.path.exists(practice_config_file):
    practice_config = ConfigList(taskname = trainname + '_Practice', exp_len = 120, seed = 1939)
    # ensure the first 40 trials are TS1 and second 40 are TS2
    practice_config.trial_states[0:40] = [0]*40; practice_config.trial_states[40:80] = [1]*40
    practice_config_file = practice_config.get_config(filey = practice_config_file)

# train 
if train_on:
    train_config = ConfigList(taskname = trainname, subjid = subject_code, exp_len = train_len, ts_order = ts_order, rp = recursive_p)
    train_config_file = train_config.get_config()
else:
    train_config_file = glob.glob('../Config_Files/*Context_' +subject_code +'*yaml')[-1]
    
test_config = ConfigList()
test_config.load_config_settings(train_config_file, taskname=train_config.taskname+'_test', exp_len=test_len)
test_config.setup_trial_list(displayFB = False)
test_config_file = test_config.get_config()

# setup tasks
practice=probContextTask(practice_config_file,subject_code, save_dir=save_dir, fullscreen = fullscr, mode = 'practice')
train=probContextTask(train_config_file,subject_code, save_dir=save_dir, fullscreen = fullscr)
test=probContextTask(test_config_file,subject_code, save_dir=save_dir, fullscreen = fullscr)

# setup bot if on
if bot_on == True:
    train.setBot(bot = test_bot(train_config_file, mode = bot_mode), mode = "full")
    test.setBot(bot = test_bot(test_config_file, mode = bot_mode), mode = "full")


# ****************************************************************************
# ************** RUN TASK ****************************************************
# ****************************************************************************

# ****************************************************************************
# Start Practice
# ****************************************************************************

if practice_on and not bot_on:
    # prepare to start
    practice.setupWindow()
    practice.defineStims()
    task_intro_text = [
        'Welcome\n\nPress 5 to move through instructions',
        """
        This experiment starts with a training phase followed by a testing phase.
        Training will last 45 minutes and testing will last 30 minutes.
        
        Your performance on the training AND test phase determines your bonus payment. 
        To perform well on the test phase you'll need to stay
        motivated and learn as much as possible in the training phase.
        """,
        """
        In the training phase, shapes will appear on the screen
        one at a time, and you will need to learn how to respond to them.
        
        Your responses will consist of one of four buttons: 'd', 'f', 'j' and 'k'.
        Use your index and middle fingers on both hands to respond.
        
        The goal is to learn the best key(s) to press for each shape.
        After you press a key, the shape will disappear and 
        you will get a point if you responded correctly.
        
        Press '5' to see the four shapes that will be used in practice.
        """,
        """
        As you could see, these shapes differ in their identity (which
        shape they are) and their color.
        
        Your responses should depend on these features. At certain points
        in the experiment you should respond based on color, and at other times
        you should respond based on identity, but not both at the same time.
        
        We will now practice responding to the shapes. For these trials,
        just pay attention to the identity of the shape when making your response.
        
        Please wait for the experimenter.
        """,
        """
        In those trials, one key worked for the pentagon, and one worked for the triangle.
        
        We'll now practice responding to the color of the shape.
        
        Please wait for the experimenter.
        """,
        """
        In those trials, one key worked for yellow shapes and one for green shapes.
        
        For the rest of the experiment, the shape's vertical position will also 
        change from trial to trial.
        
        Press "5" to see this in a few trials.
        """,
        """
        Your job in this experiment is to figure out on which trials you
        should respond based on identity and on which trials you should
        respond based on color.
        
        Use the points during the training phase to learn how to respond.
        """,
        """
        After the training phase, there will be a test phase 
        with no feedback. You will still be earning points, and these
        test phase points will also be used to determine your bonus pay.
        
        Because there is no feedback, it will be impossible to learn anything
        new during the test phase. Therefore it is important that you learn all
        you can during the training phase.
        """,
        """
        You must respond while the shape is on the screen.
        Please respond as quickly and accurately as possible.
        
        The task is hard! Stay motivated and try to learn
        all you can.
        
        We will start with a brief practice session. 
        Please wait for the experimenter.
        """
    ]
    
    for line in task_intro_text:
        practice.presentTextToWindow(line)
        resp,practice.startTime=practice.waitForKeypress(practice.trigger_key)
        practice.checkRespForQuitKey(resp)
        event.clearEvents()
        if 'used in practice' in line:
            practice.presentStims(mode = 'practice')
            resp,practice.startTime=practice.waitForKeypress(practice.trigger_key)
            practice.checkRespForQuitKey(resp)
        if "pay attention to the identity of the shape" in line:
            pos_count = 0
            startTime = core.getTime()
            for trial in practice.stimulusInfo[0:40]:
                # wait for onset time
                while core.getTime() < trial['onset'] + startTime:
                        key_response=event.getKeys(None,True)
                        if len(key_response)==0:
                            continue
                        for key,response_time in key_response:
                            if practice.quit_key==key:
                                practice.shutDownEarly()
                trial=practice.presentTrial(trial)
                if trial['FB'] == 1:
                    pos_count += 1
                else:
                    pos_count = 0
                if pos_count ==6:
                    break
            core.wait(1)
        if "responding to the color of the shape" in line:
            pos_count = 0
            startTime = core.getTime()
            elapsed_time = practice.stimulusInfo[39]['onset']+1
            for trial in practice.stimulusInfo[40:80]: 
                # wait for onset time
                while core.getTime() < trial['onset'] + startTime - elapsed_time:
                        key_response=event.getKeys(None,True)
                        if len(key_response)==0:
                            continue
                        for key,response_time in key_response:
                            if practice.quit_key==key:
                                practice.shutDownEarly()
                trial=practice.presentTrial(trial)
                if trial['FB'] == 1:
                    pos_count += 1
                else:
                    pos_count = 0
                if pos_count ==6:
                    break
            core.wait(1)
        if "see this in a few trials" in line:
            pos_count = 0
            startTime = core.getTime()
            elapsed_time = practice.stimulusInfo[79]['onset']+1
            for trial in practice.stimulusInfo[80:84]: 
                # wait for onset time
                while core.getTime() < trial['onset'] + startTime - elapsed_time:
                        key_response=event.getKeys(None,True)
                        if len(key_response)==0:
                            continue
                        for key,response_time in key_response:
                            if practice.quit_key==key:
                                practice.shutDownEarly()
                trial=practice.presentTrial(trial)
            core.wait(1)
    
    for trial in practice.stimulusInfo[84:]:
        # wait for onset time
        startTime = core.getTime()
        elapsed_time = practice.stimulusInfo[83]['onset'] + 1
        while core.getTime() < trial['onset'] + practice.startTime - elapsed_time:
                key_response=event.getKeys(None,True)
                if len(key_response)==0:
                    continue
                for key,response_time in key_response:
                    if practice.quit_key==key:
                        practice.shutDownEarly()
        trial=practice.presentTrial(trial)
    
    practice.presentTextToWindow(
    """
    That's enough practice. In the actual experiment, there will
    be new shapes that you have to learn about. You still have to
    learn when to respond based on the identity or color of the shape,
    but the correct responses may be different from what you learned
    during practice. 
    
    Before we start the experiment
    press 5 to see the shapes you will have to respond to
    during the training and test phases.
    """)
    resp,practice.startTime=practice.waitForKeypress(practice.trigger_key)
    practice.checkRespForQuitKey(resp)
    practice.presentStims(mode = 'task')
    resp,practice.startTime=practice.waitForKeypress(practice.trigger_key)
    practice.checkRespForQuitKey(resp)
    
    # clean up
    practice.closeWindow()

# ****************************************************************************
# Start training
# ****************************************************************************

if train_on:
    # prepare to start
    train.setupWindow()
    train.defineStims()
    if bot_on == False:
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

    pause_trial = train.stimulusInfo[len(train.stimulusInfo)/2]
    train.run_task(pause_trial=pause_trial)    
    
    #************************************
    # Send text about train performance
    #************************************
    if bot_on == False:   
        send_message('Training done')
        

        
# ****************************************************************************
# Start test
# ****************************************************************************
        
if test_on:
    # prepare to start
    test.setupWindow()
    test.defineStims()
    if bot_on == False:
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
        
    pause_trial = test.stimulusInfo[len(test.stimulusInfo)/2]
    test.run_task(pause_trial = pause_trial)
        
    #************************************
    # Send text about test performance
    #************************************
    if bot_on == False:   
        send_message('Testing Done')
       
#************************************
# Determine payment
#************************************
points,trials = test.getPoints()
performance = (float(points)/trials-.25)/.75
pay_bonus = round(performance*5)
print('Participant ' + subject_code + ' won ' + str(points) + ' points out of ' + str(trials) + ' trials. Bonus: $' + str(pay_bonus))

#open post-task questionnaire
if bot_on == False:
    webbrowser.open_new('https://stanforduniversity.qualtrics.com/SE/?SID=SV_9KzEWE7l4xuORIF')






