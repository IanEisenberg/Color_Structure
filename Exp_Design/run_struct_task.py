"""
runcolorStructTask
"""

from psychopy import core, event
import smtplib
import json
import webbrowser
from color_struct_task_2dims import colorStructTask
from make_config_2dims import makeConfigList, makePracticeConfigList
from test_bot_2dims import test_bot
#set-up some variables

verbose=True
fullscr= True
subdata=[]
practice_on = True
task_on = True
test_on = True
bot_on = False
bot_mode = "ignore_base" #other for optimal

# set things up for practice, training and tests
try:
    f = open('IDs.txt','r')
    lines = f.readlines()
    f.close()
    last_id = lines[-1][:-1]
    subject_code = raw_input('Last subject: "%s". Input new subject code: ' % last_id);
except IOError:
    subject_code = raw_input('Input first subject code: ');
f = open('IDs.txt', 'a')
f.write(subject_code + '\n')
f.close()

train_mins = 10 #train_length in minutes
test_mins = 10 #test_length in minutes
avg_test_trial_len = 2.25 #in seconds
avg_task_trial_len = avg_test_trial_len + 1 #factor in FB
#Find the minimum even number of blocks to last at least train_length minutes
task_len = int(round(train_mins*60/avg_task_trial_len/4)*4)
test_len = int(round(test_mins*60/avg_test_trial_len/4)*4)


#set up config files
practice_config_file = '../Config_Files/Color_Struct_Practice_config.npy'
task_config_file = makeConfigList(iden = subject_code, exp_len = task_len, recursive_p = .9)

try:
    practice=colorStructTask(practice_config_file,subject_code, fullscreen = fullscr, mode = 'practice')
except SystemExit:
    practice_config_file = makePracticeConfigList()
    practice=colorStructTask(practice_config_file,subject_code, fullscreen = fullscr, mode = 'practice')

task=colorStructTask(task_config_file,subject_code, fullscreen = fullscr)
if bot_on == True:
    task.setBot(bot = test_bot(task_config_file, mode = bot_mode), mode = "short")
task.writeToLog(task.toJSON())


#************************************
# Start Practice
#************************************

if practice_on:
    # prepare to start
    practice.setupWindow()
    practice.defineStims()
    task_intro_text = [
        'Welcome\n\nPress 5 to move through instructions',
        """
        This experiment starts with a training phase followed by a testing phase,
        each lasting about 25 minutes.
        
        Your performance on the test phase determines your bonus payment
        (up to $5). To perform well on the test phase you'll need to stay
        motivated and learn as much as possible in the training phase.
        """,
        """
        In the training phase, shapes will appear on the screen
        one at a time, and you will need to learn how to respond to them.
        
        Your responses will consist of one of four buttons: 'd', 'f', 'j' and 'k'.
        Use your index and middle fingers on both hands to respond.
        
        The goal is to learn the best key(s) to press for each shape.
        """,
        """
        The shape's vertical position will also be changing on each trial.
        
        Critically, the best key to press in response to each shape 
        depends on the shape's position. 
        
        The different positions where the shapes will be presented 
        are shown on the next screen.
        """,
        """
        After you press a key, the shape will disappear and 
        you will get points indicating how well you did.
        
        After the training phase, there will be a test phase 
        with no feedback. You will still be earning points, and these
        test phase points will be used to determine your bonus pay.
        
        It is therefore very important that you use the points 
        in the training phase to learn how to best respond to each shape 
        depending on the vertical position.
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
        if 'position' in line:
            practice.presentContexts()
            resp,practice.startTime=practice.waitForKeypress(practice.trigger_key)
            practice.checkRespForQuitKey(resp)

    
    
    for trial in practice.stimulusInfo:
        # wait for onset time
        while core.getTime() < trial['onset'] + practice.startTime:
                key_response=event.getKeys(None,True)
                if len(key_response)==0:
                    continue
                for key,response_time in key_response:
                    if practice.quit_key==key:
                        practice.shutDownEarly()
        trial=practice.presentTrial(trial)
    
    # clean up
    practice.closeWindow()

#************************************
# Start training
#************************************

if task_on:
    # prepare to start
    task.setupWindow()
    task.defineStims()
    if bot_on == False:
        task.presentTextToWindow(
            """
            We will now start the experiment.
            
            There will be one break half way through. 
            
            Please wait for the experimenter.
            """)
        resp,task.startTime=task.waitForKeypress(task.trigger_key)
        task.checkRespForQuitKey(resp)
        event.clearEvents()
    else:
        task.startTime = core.getTime()
        
    
    pause_trial = task.stimulusInfo[len(task.stimulusInfo)/2]
    pause_time = 0
    for trial in task.stimulusInfo:
        if not task.bot:
            if trial == pause_trial:
                time1 = core.getTime()
                task.presentTextToWindow("Take a break! Press '5' when you're ready to continue.")
                task.waitForKeypress(task.trigger_key)
                task.clearWindow()
                pause_time = core.getTime() - time1
        
        #if botMode = short, don't wait for onset times
        if task.botMode != 'short':
            # wait for onset time
            while core.getTime() < trial['onset'] + task.startTime + pause_time:
                    key_response=event.getKeys(None,True)
                    if len(key_response)==0:
                        continue
                    for key,response_time in key_response:
                        if task.quit_key==key:
                            task.shutDownEarly()
                        elif task.trigger_key==key:
                            task.trigger_times.append(response_time-task.startTime)
                            task.waitForKeypress()
                            continue
    
        trial=task.presentTrial(trial)
        task.writeToLog(json.dumps(trial))
        task.alldata.append(trial)
        #print('state = ' + str(trial['state'])+ ', value: ' + str(np.mean(trial['context']))) 
        
    
    task.writeToLog(json.dumps({'trigger_times':task.trigger_times}))
    task.writeData()
    if bot_on == False:
        task.presentTextToWindow('Thank you. Please wait for the experimenter.')
        task.waitForKeypress(task.quit_key)


    # clean up
    task.closeWindow()

#************************************
# Send text about task performance
#************************************
    if bot_on == False:   
        username = "thedummyspeaks@gmail.com"
        password = "r*kO84gSzzD4"
        
        atttext = "9148155478@txt.att.net"
        message = "Training done. Points " + str(task.getPoints())
        
        msg = """From: %s
        To: %s
        Subject: text-message
        %s""" % (username, atttext, message)
        
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.starttls()
        server.login(username,password)
        server.sendmail(username, atttext, msg)
        server.quit()
    
#************************************
# Start test
#************************************

if test_on:
    
    test_config_file = makeConfigList(taskname = 'Color_Struct_noFB', iden = subject_code, exp_len = test_len, 
                                      recursive_p = .9, FBDuration = 0, FBonset = 0, action_keys = task.getActions(),
                                      ts_order = task.getTSorder())
    test=colorStructTask(test_config_file,subject_code, fullscreen = fullscr)
    if bot_on == True:
        test.setBot(bot = test_bot(test_config_file, mode = bot_mode), mode = "short")

    test.writeToLog(test.toJSON())
    
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
                            
        resp,test.startTime=test.waitForKeypress(test.trigger_key)
        test.checkRespForQuitKey(resp)
        event.clearEvents()
    else:
        test.startTime = core.getTime()
        
    pause_trial = test.stimulusInfo[len(test.stimulusInfo)/2]
    pause_time = 0
    for trial in test.stimulusInfo:
        if not test.bot:
            if trial == pause_trial:
                time1 = core.getTime()
                test.presentTextToWindow("Take a break! Press '5' when you're ready to continue.")
                test.waitForKeypress(test.trigger_key)
                test.clearWindow()
                pause_time = core.getTime() - time1
            
        #if botMode = short, don't wait for onset times
        if test.botMode != 'short':
            # wait for onset time
            while core.getTime() < trial['onset'] + test.startTime:
                    key_response=event.getKeys(None,True)
                    if len(key_response)==0:
                        continue
                    for key,response_time in key_response:
                        if test.quit_key==key:
                            test.shutDownEarly()
                        elif test.trigger_key==key:
                            test.trigger_times.append(response_time-test.startTime)
                            continue
    
        trial=test.presentTrial(trial)
        test.writeToLog(json.dumps(trial))
        test.alldata.append(trial)
    
    test.writeToLog(json.dumps({'trigger_times':task.trigger_times}))
    test.writeData()
    if bot_on == False:
        test.presentTextToWindow('Thank you. Please wait for the experimenter.')
        test.waitForKeypress(task.quit_key)
    
    # clean up
    test.closeWindow()
    

#************************************
# Send text about test performance
#************************************   
    if bot_on == False:
        username = "thedummyspeaks@gmail.com"
        password = "r*kO84gSzzD4"
        
        atttext = "9148155478@txt.att.net"
        message = "Test done. Points " + str(test.getPoints())
        
        msg = """From: %s
        To: %s
        Subject: text-message
        %s""" % (username, atttext, message)
        
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.starttls()
        server.login(username,password)
        server.sendmail(username, atttext, msg)
        server.quit()
    
    
#************************************
# Determine payment
#************************************
points,trials = test.getPoints()
performance = float(points)/trials
pay_bonus = round(performance*5*2)/2.0
print('Participant ' + subject_code + ' won ' + str(points) + ' points out of ' + str(trials) + ' trials. Bonus: $' + str(pay_bonus))
if bot_on == False:
    webbrowser.open_new('https://stanforduniversity.qualtrics.com/SE/?SID=SV_aV1hwNrNXgX5NYN')






