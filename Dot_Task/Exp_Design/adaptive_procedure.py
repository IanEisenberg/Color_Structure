"""
generic task using psychopy
"""

import cPickle
import datetime
import json
import numpy as np
from psychopy import visual, core, event
from psychopy.data import QuestHandler, StairHandler
import subprocess
import sys,os
import yaml

from flowstim import OpticFlow
from utils import pixel_lab2rgb

class adaptiveThreshold:
    """ class defining a probabilistic context task
    """
    
    def __init__(self,config_file,subjid,save_dir,verbose=True, 
                 fullscreen = False, mode = 'task', trackers = None):
        # set up some variables
        self.stimulusInfo=[]
        self.loadedStimulusFile=[]
        self.startTime=[]
        self.alldata=[]
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        #looks up the hash of the most recent git push. Stored in log file
        self.gitHash = subprocess.check_output(['git','rev-parse','--short','HEAD'])[:-1]
        # load config file
        self.config_file=config_file
        try:
            self.loadConfigFile(config_file)
        except:
            print(mode + ': cannot load config file')
            sys.exit()
            
        self.save_dir = save_dir  
        self.subjid=subjid
        # set up window
        self.win=[]
        self.window_dims=[800,600]
        self.aperture=None
        
        self.textStim=[]
        self.trialnum = 0
        self.track_response = []
        self.fullscreen = fullscreen
        self.text_color = [1]*3
        self.pointtracker = 0
        #Choose 'practice', 'task': determines stimulus set to use
        self.mode = mode
        # set up recording files
        self.logfilename='%s_%s_%s.log'%(self.subjid,self.taskname,self.timestamp)
        self.datafilename='%s_%s_%s.pkl'%(self.subjid,self.taskname,self.timestamp)
        # log initial state
        self.writeToLog(self.toJSON())
        # convert colors to array to be more useable
        self.stim_colors = np.array(self.stim_colors)
        # setup trackers
        if trackers is None:
            trackers = {}
        self.defineTrackers(trackers)
    
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
    
    def writeToLog(self,msg):
        f=open(os.path.join(self.save_dir,'Log',self.logfilename),'a')
        f.write(msg)
        f.write('\n')
        f.close()
         
    def writeData(self):
        save_loc = os.path.join(self.save_dir,'RawData',self.datafilename)
        data = {}
        data['taskinfo']=self.taskinfo
        data['configfile']=self.config_file
        data['subcode']=self.subjid
        data['timestamp']=self.timestamp
        data['taskdata']=self.alldata
        data['trackers'] = self.trackers
        f=open(save_loc,'w')
        cPickle.dump(data,f)
    
    #**************************************************************************
    # ******* Display Functions **************
    #**************************************************************************
    
    def setupWindow(self, aperture=True):
        """ set up the main window
        """
        self.win = visual.Window(self.window_dims, allowGUI=False, 
                                 fullscr=self.fullscreen, monitor='testMonitor', 
                                 units='norm', allowStencil=True,
                                 color=[-1,-1,-1])   
        if aperture==True:
            # define aperture
            aperture_size = 1.5
            aperture_vertices = visual.Aperture(self.win, size=aperture_size, units='norm').vertices
            ratio = float(self.win.size[1])/self.win.size[0]
            aperture_vertices[:,0]*=ratio
            self.aperture = visual.Aperture(self.win, size=aperture_size, units='norm', shape = aperture_vertices)
            self.aperture.disable()
                     
        self.win.flip()
        self.win.flip()
        
        
    def presentTextToWindow(self,text,size=.15):
        """ present a text message to the screen
        return:  time of completion
        """
        
        if not self.textStim:
            self.textStim=visual.TextStim(self.win, text=text,font='BiauKai',
                                height=size,color=self.text_color, colorSpace=u'rgb',
                                opacity=1,depth=0.0,
                                alignHoriz='center',wrapWidth=50)
        else:
            self.textStim.setText(text)
            self.textStim.setHeight(size)
            self.textStim.setColor(self.text_color)
        self.textStim.draw()
        self.win.flip()
        return core.getTime()

    def clearWindow(self):
        """ clear the main window
        """
        if self.textStim:
            self.textStim.setText('')
            self.win.flip()
        else:
            self.presentTextToWindow('')

    def waitForKeypress(self,key=[]):
        """ wait for a keypress and return the pressed key
        - this is primarily for waiting to start a task
        - use getResponse to get responses on a task
        """
        start=False
        event.clearEvents()
        while start==False:
            key_response=event.getKeys()
            if len(key_response)>0:
                if key:
                    if key in key_response or self.quit_key in key_response:
                        start=True
                else:
                    start=True
        self.clearWindow()
        return key_response,core.getTime()
        
    def closeWindow(self):
        """ close the main window
        """
        if self.win:
            self.win.close()

    def checkRespForQuitKey(self,resp):
        if self.quit_key in resp:
            self.shutDownEarly()

    def shutDownEarly(self):
        self.closeWindow()
        sys.exit()
    
    def getPastAcc(self, time_win):
        """Returns the ratio of hits/trials in a predefined window
        """
        if time_win > self.trialnum:
            time_win = self.trialnum
        return sum(self.track_response[-time_win:])
        
    def getStims(self):
        return self.stims
        
    def getActions(self):
        return self.action_keys
        
    def getTSorder(self):
        return [self.taskinfo['states'][0]['ts'],
                self.taskinfo['states'][1]['ts']]
        
    def getPoints(self):
        return (self.pointtracker,self.trialnum)
        
    def defineStims(self, stim = None):
        height = .02
        ratio = self.win.size[1]/float(self.win.size[0])
        if stim == None:
            self.stim=OpticFlow(self.win, speed=self.base_speed,
                                color=[0,0,0], nElements = 4000,
                                sizes=[height*ratio, height])
        else:
            self.stim = stim 
    
    def defineTrackers(self, trackers, method='quest'):
        if self.ts == "motion":
            difficulties = self.motion_difficulties
            maxVal = self.base_speed
        elif self.ts == "color":
            difficulties = self.color_difficulties
            maxVal = self.color_starts[0]
        if method=='basic':
            step_lookup = {'easy':5,
                           'medium': 3,
                           'hard': 2}
            for key,val in difficulties.items():
                nDown = step_lookup[key]
                trackers[key] = StairHandler(startVal=val, minVal=0, 
                                            maxVal=maxVal,
                                            stepSizes=maxVal/10.0, 
                                            stepType='lin',
                                            nDown=nDown,
                                            nReversals=20,
                                            staircase=trackers.get(key,None))
        elif method=='quest':
            quest_lookup = {'easy': .85,
                           'medium': .75,
                           'hard': .65}
            for key,val in difficulties.items():
                threshold = quest_lookup[key]
                trackers[key] = QuestHandler(pThreshold=threshold,
                                            nTrials = self.exp_len,
                                            startVal=val, startValSd=maxVal/2,
                                            minVal=0.0001, 
                                            maxVal=maxVal,
                                            gamma=.01,
                                            beta=3.5,
                                            staircase=trackers.get(key,None))
        self.trackers = trackers
        
        
    def getTrialAttributes(self,stim):
        ss, sd, cs, cd, md, color = [stim[k] for k in 
                                     ['speedStrength','speedDirection',
                                      'colorStrength','colorDirection',
                                      'motionDirection', 'colorSpace']]
        # transform word difficulties into numbers
        ss = self.motion_difficulties[ss]
        cs = self.color_difficulties[cs]
        # create start and end points
        speed_start = self.base_speed
        speed_end = self.base_speed + ss*sd
        
        color1_start = color
        color1_end = color1_start+cd*cs
        colors = [self.stim_colors[0]*color1_start + 
                self.stim_colors[1]*(1-color1_start),
                self.stim_colors[0]*color1_end + 
                self.stim_colors[1]*(1-color1_end)]
        color_start,color_end = colors
        return [speed_start, speed_end, color_start, color_end, md]
                                      
        
    def presentStim(self, trial_attributes, duration=.5, response_window=1,
                    mode = 'practice', clock=True):
        """ Used during instructions to present possible stims
        """
        ss,se,cs,ce,md = trial_attributes
        cs = np.array(cs)
        ce = np.array(ce)
        # reset dot position
        self.stim.setupDots()
        if mode == 'practice':
            self.stim.updateTrialAttributes(dir=md,color=cs,speed=ss)

        elif mode == 'task':
            self.stim.updateTrialAttributes(dir=md,color=cs,speed=ss)
            
        stim_clock = core.Clock()
        recorded_keys = []
        if self.aperture: self.aperture.enable()
        while stim_clock.getTime() < duration+response_window:
            if stim_clock.getTime() < duration:
                percent_complete = stim_clock.getTime()/duration
                # smoothly move color over the duration
                color = cs*(1-percent_complete) + ce*percent_complete
                # change speed
                speed = ss*(1-percent_complete) + se*percent_complete
                # convert to rgb
                color = pixel_lab2rgb(color)
                self.stim.updateTrialAttributes(color=color, speed=speed)
                self.stim.draw()
            elif 0<(stim_clock.getTime()-duration)<.05:
                self.win.flip()
                self.win.flip()
            keys = event.getKeys(self.action_keys + [self.quit_key],
                                 timeStamped=clock)
            for key,response_time in keys:
                # check for quit key
                if key == self.quit_key:
                    self.shutDownEarly()
                recorded_keys+=keys
        if self.aperture: self.aperture.disable()
        self.win.flip(clearBuffer=True)
        return recorded_keys
            
    def getCorrectChoice(self,trial_attributes,ts):
        ss,se,cs,ce,md = trial_attributes
        # action keys are set up as the choices for ts1 followed by ts2
        # so the index for the correct choice must take that into account
        if ts == 'motion':
            correct_choice = int(se>ss)
        elif ts == 'color':
            # correct choice is based on whether the color became "more extreme"
            # i.e. more green/red
            correct_choice = int(abs(cs[1])>abs(ce[1]))+2
        return self.action_keys[correct_choice]
        
    def presentTrial(self,trial):
        """
        This function presents a stimuli, waits for a response, tracks the
        response and RT and presents appropriate feedback. This function also controls the timing of FB 
        presentation.
        """
        trialClock = core.Clock()
        self.trialnum += 1
        stim = trial['stim']
        # update difficulties based on adaptive tracker
        if self.ts == "motion":
            strength = stim["speedStrength"]
            difficulties = self.motion_difficulties
        elif self.ts == "color":
            strength = stim["colorStrength"]
            difficulties = self.color_difficulties
        tracker = self.trackers[strength]
        decision_var = tracker.next()
        difficulties[strength] = decision_var
        trial['decision_var'] = decision_var
        # get stim attributes
        trial_attributes = self.getTrialAttributes(stim)
        print('*'*40)
        print('Speed Change: ', trial_attributes[0:2]) 
        print('Taskset: %s, choice value: %s\nSpeed: %s, Strength: %s \
              \nColorDirection: %s, ColorStrength: %s \
              \nCorrectChoice: %s' % 
              (trial['ts'], decision_var, 
               stim['speedDirection'], stim['speedStrength'], 
               stim['colorDirection'],stim['colorStrength'],
               self.getCorrectChoice(trial_attributes,trial['ts'])))
        
        
        trial['actualOnsetTime']=core.getTime() - self.startTime
        trial['stimulusCleared']=0
        trial['response'] = 999
        trial['rt'] = 999
        trial['FB'] = 999
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
            if correct_choice == choice:
                FB = trial['reward_amount']
                tracker.addResponse(1)
            else:
                FB = trial['punishment_amount']
                tracker.addResponse(0)
            # add current tracker estimate
            trial['quest_estimate'] = tracker.mean()
            # record points for bonus
            self.pointtracker += FB
            # Present FB to window
            if trial['displayFB'] == True:
                trial['FB'] = FB
                core.wait(trial['FBonset'])  
                trial['actualFBOnsetTime'] = trialClock.getTime()-trial['stimulusCleared']
                if FB == 1:
                    self.presentTextToWindow('+1 point')
                else:
                    self.presentTextToWindow('+' + str(FB) + ' points')
                core.wait(trial['FBDuration'])
                self.clearWindow()        
        # If subject did not respond within the stimulus window clear the stim
        # and admonish the subject
        if trial['rt']==999:
            tracker.addResponse(0)
            self.clearWindow()            
            core.wait(trial['FBonset'])
            self.presentTextToWindow('Please Respond Faster')
            core.wait(trial['FBDuration'])
            self.clearWindow()
        
        # log trial and add to data
        self.writeToLog(json.dumps(trial))
        self.alldata.append(trial)
        return trial
            
    def run_task(self, pause_trials = None):
        self.startTime = core.getTime()
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
        self.writeData()
        self.presentTextToWindow('Thank you. Please wait for the experimenter',
                                 size=.05)
        self.waitForKeypress(self.quit_key)
        self.closeWindow()
















