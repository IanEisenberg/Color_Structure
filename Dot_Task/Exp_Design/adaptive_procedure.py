"""
adaptive_procedure.py
---------------------
Handles all the highlevel functions during the adaptive (calibration)
phase of the experiment. 

"""
from BaseExp import BaseExp
import json
import numpy as np
from psychopy import visual, core, event
from psychopy.data import QuestHandler, StairHandler
import subprocess
import sys,os
import yaml
from flowstim import OpticFlow

class adaptiveThreshold(BaseExp):
    """ class defining a probabilistic context task
    """
    
    def __init__(self,config_file,subjid,save_dir,verbose=True, 
                 fullscreen=False, mode='task', trackers=None):
        # set up some variables
        self.stimulusInfo=[]
        self.loadedStimulusFile=[]
        self.startTime=[]
        self.alldata=[]
        #looks up the hash of the most recent git push. Stored in log file
        self.gitHash = subprocess.check_output(['git','rev-parse','--short','HEAD'])[:-1]
        # load config file
        self.config_file=config_file
        try:
            self.loadConfigFile(config_file)
        except:
            print(mode + ': cannot load config file')
            sys.exit()
            
        self.aperture=None
        self.trialnum = 0
        self.track_response = []
        self.pointtracker = 0
        #Choose 'practice', 'task': determines stimulus set to use
        self.mode = mode
        # setup trackers
        if trackers is None:
            trackers = {}
        self.defineTrackers(trackers)
        # init Base Exp
        super(adaptiveThreshold, self).__init__(self.taskname, subjid, save_dir, fullscreen)
    
    def loadConfigFile(self,filename):
        """ load a config file from yaml
        """
        if not os.path.exists(filename):
            raise BaseException('Config file not found')
        config_file = yaml.load(open(filename,'r'))
        for trial in config_file:
            if trial.has_key('taskname'):
                self.taskinfo=trial
                for k in self.taskinfo.iterkeys():
                    self.__dict__[k]=self.taskinfo[k]
            else:
                self.stimulusInfo.append(trial)
        if len(self.stimulusInfo)>0:
            self.loadedStimulusFile=filename
            
    #**************************************************************************
    # ******* Function to Save Data **************
    #**************************************************************************
    
    def toJSON(self):
        """ log the initial conditions for the task. Exclude the list of all
        trials (stimulusinfo), the bot, and taskinfo (self.__dict__ includes 
        all of the same information as taskinfo)
        """
        init_dict = {k:self.__dict__[k] for k in self.__dict__.iterkeys() if k 
                    not in ('clock', 'stimulusInfo', 'alldata', 'bot', 'taskinfo','win')}
        return json.dumps(init_dict)
    
    #**************************************************************************
    # ******* Display Functions **************
    #**************************************************************************
    def defineStims(self, stim = None):
        height = 1
        ratio = .3
        if stim == None:
            self.stim = OpticFlow(self.win, 
                                 speed=self.base_speed,
                                 color=[1,1,1], 
                                 nElements = 2000,
                                 sizes=[height*ratio, height])
        else:
            self.stim = stim 
        # define fixation
        self.fixation = self.stim.fixation
    
    def defineTrackers(self, trackers, method='quest'):
        if self.ts == "motion":
            difficulties = self.motion_difficulties
            maxVal = self.base_speed
        elif self.ts == "orientation":
            difficulties = self.ori_difficulties
            maxVal = 25 # no more than a 20 degree change
        if method=='basic':
            step_lookup = {'easy':5,
                           'hard': 3}
            for (pedestal, difficulty),val in difficulties.items():
                key = (pedestal,difficulty)
                nDown = step_lookup[difficulty]
                trackers[key] = StairHandler(startVal=val, minVal=0, 
                                            maxVal=maxVal,
                                            stepSizes=maxVal/10.0, 
                                            stepType='lin',
                                            nDown=nDown,
                                            nReversals=20,
                                            staircase=trackers.get(key,None))
        elif method=='quest':
            quest_lookup = {'easy': .85,
                            'hard': .7}
            for (pedestal, difficulty), val in difficulties.items():
                key = (pedestal,difficulty)
                threshold = quest_lookup[difficulty]
                trackers[key] = QuestHandler(pThreshold=threshold,
                                            nTrials = self.exp_len,
                                            startVal=val, startValSd=maxVal/2,
                                            minVal=0.0001, 
                                            maxVal=maxVal,
                                            gamma=.01,
                                            grain=maxVal/400.0,
                                            range=maxVal*2,
                                            beta=3.5,
                                            staircase=trackers.get(key,None))
        self.trackers = trackers
        
    def presentTrial(self,trial, mode = 'trial', show_FB = True):
        """
        This function presents a stimuli, waits for a response, tracks the
        response and RT and presents appropriate feedback. 
        This function also controls the timing of FB 
        presentation.
        -------------
        modes: 'trial' (default) saves values to tracker
                'preactice' does not save values to tracker
        """
        trialClock = core.Clock()
        if mode == 'trial':
            self.trialnum += 1
        stim = trial['stim']
        # update difficulties based on adaptive tracker
        if self.ts == "motion":
            strength = stim["speedStrength"]
            pedestal = stim["motionDirection"]
            difficulties = self.motion_difficulties
        elif self.ts == "orientation":
            strength = stim["oriStrength"]
            pedestal = stim["oriBase"]
            difficulties = self.ori_difficulties
        tracker_key = (pedestal,strength)
        tracker = self.trackers[tracker_key]
        decision_var = tracker.next()
        difficulties[(pedestal,strength)] = decision_var
        trial['decision_var'] = decision_var
        # get stim attributes
        trial_attributes = self.getTrialAttributes(stim)
        print('*'*40)
        print('Tracker: %s' % str(tracker_key), 'Best Guess: %s' % tracker.mean()) 
        print('Taskset: %s, choice value: %s\nSpeed: %s, Strength: %s \
              \nOriDirection: %s, OriStrength: %s \
              \nCorrectChoice: %s' % 
              (trial['ts'], decision_var, 
               stim['speedDirection'], stim['speedStrength'], 
               stim['oriDirection'],stim['oriStrength'],
               self.getCorrectChoice(trial_attributes,trial['ts'])))
        
        
        trial['actualOnsetTime']=core.getTime() - self.startTime
        trial['response'] = np.nan
        trial['rt'] = np.nan
        trial['FB'] = np.nan
        # present stimulus and get response
        event.clearEvents()
        trialClock.reset()
        keys = self.presentStim(trial_attributes, trial['stimulusDuration'], 
                                trial['responseWindow'], mode = 'task',
                                clock=trialClock)
        if len(keys)>0:
            choice = keys[0][0]
            print('Choice: %s' % choice)
            # record response
            trial['response'] = choice
            trial['rt'] = keys[0][1]
            # record any responses after the first
            trial['secondary_responses']=[i[0] for i in keys[1:]]
            trial['secondary_rts']=[i[1] for i in keys[1:]]
            # get feedback and update tracker
            correct_choice = self.getCorrectChoice(trial_attributes,trial['ts'])
            #update tracker if in trial mode
            if correct_choice == choice:
                FB = trial['reward_amount']
                if mode == 'trial':
                    tracker.addResponse(1)
            else:
                FB = trial['punishment_amount']
                if mode == 'trial':
                    tracker.addResponse(0)
            # add current tracker estimate
            trial['quest_estimate'] = tracker.mean()
            # record points for bonus
            if mode == 'trial':
                self.pointtracker += FB
            # Present FB to window
            if trial['displayFB'] == True:
                trial['FB'] = FB
                core.wait(trial['FBonset'])  
                trial['actualFBOnsetTime'] = trialClock.getTime()
                if FB == 1 and show_FB:
                    self.presentTextToWindow('CORRECT')
                elif show_FB:
                    self.presentTextToWindow('INCORRECT')
                core.wait(trial['FBDuration'])
                self.clearWindow(fixation=self.fixation)        
        # If subject did not respond within the stimulus window clear the stim
        # and admonish the subject
        else:
            if mode == 'trial':
                tracker.addResponse(0)
            self.clearWindow()            
            core.wait(trial['FBonset'])
            if show_FB:
                self.presentTextToWindow('Please Respond Faster')
            core.wait(trial['FBDuration'])
            self.clearWindow(fixation=self.fixation)
        
        # log trial and add to data
        self.writeToLog(json.dumps(trial))
        self.alldata.append(trial)
        return trial
    
    

    def run_task(self):
        self.setupWindow()
        self.defineStims()
        
        # present intro screen
        self.presentInstruction(self.ts.title(), size=.15)
        self.startTime = core.getTime()
        # set up pause trials
        pause_trials = np.round(np.linspace(0,self.exp_len,3))[1:-1]
        pause_time = 0
        if pause_trials is None: pause_trials = []
        for trial in self.stimulusInfo:
            if trial['trial_count'] in pause_trials:
                time = core.getTime()
                self.presentTextToWindow("Take a break! Press '5' when you're ready to continue.", size = .1)
                self.waitForKeypress(self.trigger_key)
                self.clearWindow()
                self.aperture.enable()
                pause_time += core.getTime() - time
            
            # wait for onset time
            while core.getTime() < trial['onset'] + self.startTime + pause_time:
                    key_response=event.getKeys(None,True)
                    if len(key_response)==0:
                        continue
                    for key,response_time in key_response:
                        if self.quit_key==key:
                            self.shutDownEarly()
            self.presentTrial(trial)
                
        # clean up and save
        other_data={'taskinfo': self.taskinfo,
                    'configfile': self.config_file,
                    'trackers': self.trackers}
        self.writeData(taskdata=self.alldata,
                       other_data=other_data)
        self.presentInstruction(
            """
            Thank you. Please wait for the experimenter.
            """)
        self.closeWindow()
        
    
    def run_practice(self, num_practice=10):
        self.setupWindow()
        self.defineStims()
        self.presentInstruction(
            """
            Press 5 to move through instructions                         
            """)
        self.presentInstruction(
            """
            We are going to do some practice trials.
            
            In this task, shapes will appear on the screen moving towards or away 
            from the center of the screen, while rotating clockwise or counter-clockwise.
            
            Press 5 to look at a demo. Notice that at the end of the trial 
            the cross at the center of the screen will turn green.             
            """)
        self.startTime = core.getTime()
        trial = self.stimulusInfo[0]
        while core.getTime() < trial['onset'] + self.startTime:
                key_response=event.getKeys(None,True)
                if len(key_response)==0:
                    continue
                for key,response_time in key_response:
                    if self.quit_key==key:
                        self.shutDownEarly()
        self.presentTrial(trial, mode='practice', show_FB = False)
        if self.ts == "motion":
            self.presentInstruction(
            """
            Your task is to respond to what you see on the screen.
            
            If the shapes are speeding up (regardless of direction) press "UP" on the arrow keys.
            
            If they are slowing down press "DOWN" on the arrow keys.
            
            Respond only after the cross at the center of the screen has changed to green.
            
            Please press 5 for some practice.
            
            """)
        elif self.ts == "orientation":
                        self.presentInstruction("""
            Your task is to respond to what you see on the screen.
            
            If the shapes are rotating clockwise press "LEFT" on the arrow keys.
            
            If they rotating counter-clockwise press "RIGHT" on the arrow keys.
            
            Respond only after the cross at the center of the screen has changed to green.
            
            Please press 5 for some practice.
            """)
        self.startTime = core.getTime()
        trial_timing = self.stimulusInfo[0:num_practice]#get timing from the first few trials
        practice = np.random.choice(self.stimulusInfo, num_practice) #get random trials
        for num, trial in enumerate(practice):            
            # wait for onset time
            while core.getTime() < trial_timing[num]['onset'] + self.startTime:
                    key_response=event.getKeys(None,True)
                    if len(key_response)==0:
                        continue
                    for key,response_time in key_response:
                        if self.quit_key==key:
                            self.shutDownEarly()
            self.presentTrial(trial, mode='practice')
        self.presentInstruction(
            """
            Press 5 to move on to the testing phase.
            """)
        self.closeWindow()

        #for _ in range(self.num_practice_trials):
          #  trial = np.random.choice(self.stimulusInfo)
         #   self.presentTrial(trial, mode='practice')














