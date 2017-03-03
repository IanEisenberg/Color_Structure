"""
generic task using psychopy
"""

import datetime
import json
import numpy as np
from psychopy import visual, core, event
import subprocess
import sys,os
import yaml

from flowstim import OpticFlow
from utils import pixel_lab2rgb

class adaptiveThreshold:
    """ class defining a probabilistic context task
    """
    
    def __init__(self,config_file,subjid,save_dir,verbose=True, 
                 fullscreen = False, mode = 'task'):
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
        self.datafilename='%s_%s_%s.yaml'%(self.subjid,self.taskname,self.timestamp)
        # log initial state
        self.writeToLog(self.toJSON())
    
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
        f=open(save_loc,'w')
        yaml.dump(data,f)
    
    #**************************************************************************
    # ******* Display Functions **************
    #**************************************************************************
    
    def setupWindow(self):
        """ set up the main window
        """
        self.win = visual.Window(self.window_dims, allowGUI=False, 
                                 fullscr=self.fullscreen, monitor='testMonitor', 
                                 units='norm', allowStencil=True,
                                 color=[-1,-1,-1])                        
        self.win.flip()
        self.win.flip()
        
    def presentTextToWindow(self,text):
        """ present a text message to the screen
        return:  time of completion
        """
        
        if not self.textStim:
            self.textStim=visual.TextStim(self.win, text=text,font='BiauKai',
                                height=.15,color=self.text_color, colorSpace=u'rgb',
                                opacity=1,depth=0.0,
                                alignHoriz='center',wrapWidth=50)
            self.textStim.setAutoDraw(True) #automatically draw every frame
        else:
            self.textStim.setText(text)
            self.textStim.setColor(self.text_color)
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
            self.stim=OpticFlow(self.win, speed=.02,
                                color=[0,0,0], nElements = 3000,
                                sizes=[height*ratio, height])
        else:
            self.stim = stim 
        
    def presentStim(self, stim, duration = .5, mode = 'practice'):
        """ Used during instructions to present possible stims
        """
        ss,se,cs,ce,md = [stim[k] for k in ['speedStart','speedEnd',
                       'colorStart','colorEnd','motionDirection']]
        cs = np.array(cs)
        ce = np.array(ce)
        if mode == 'practice':
            self.stim.updateTrialAttributes(dir=md,color=cs,speed=ss)

        elif mode == 'task':
            self.stim.updateTrialAttributes(dir=md,color=cs,speed=ss)
            
        stim_clock = core.Clock()
        recorded_keys = []
        while stim_clock.getTime() < duration:
            percent_complete = stim_clock.getTime()/duration
            # smoothly move color over the duration
            color = cs*(1-percent_complete) + ce*percent_complete
            # change speed
            speed = ss*(1-percent_complete) + se*percent_complete
            # convert to rgb
            color = pixel_lab2rgb(color)
            self.stim.updateTrialAttributes(color=color, speed=speed)
            self.stim.draw()
            keys = event.getKeys(self.action_keys + [self.quit_key],True)
            for key,response_time in keys:
                # check for quit key
                if key == self.quit_key:
                    self.shutDownEarly()
                recorded_keys+=keys
        self.win.flip()
        self.win.flip()
        return recorded_keys
            
    def getCorrectChoice(self,stim,ts):
        # action keys are set up as the choices for ts1 followed by ts2
        # so the index for the correct choice must take that into account
        if ts == 'motion':
            correct_choice = int(bool(stim['speedDirection']+1))
        elif ts == 'color':
            # correct choice is based on whether the color became "more extreme"
            # i.e. more green/red
            correct_choice = abs(stim['colorStart'][1])>abs(stim['colorEnd'][1])+2
        return correct_choice
        
    def presentTrial(self,trial):
        """
        This function presents a stimuli, waits for a response, tracks the
        response and RT and presents appropriate feedback. This function also controls the timing of FB 
        presentation.
        """
        trialClock = core.Clock()
        self.trialnum += 1
        stim = trial['stim']
        
        print('Taskset: %s\nSpeed: %s, Strength: %s\nColorDirection: %s, ColorStrength: %s \
              \ncolorStart: %s\ncolorEnd: %s\nCorrectChoice: %s\n' % 
              (trial['ts'], stim['speedDirection'], stim['speedStrength'], 
               stim['colorDirection'],stim['colorStrength'],
               np.round(stim['colorStart'],2),np.round(stim['colorEnd'],2),
               self.getCorrectChoice(stim,trial['ts'])))
        
        
        trial['actualOnsetTime']=core.getTime() - self.startTime
        trial['stimulusCleared']=0
        trial['response'] = 999
        trial['rt'] = 999
        trial['FB'] = []
        # present stimulus and get response
        event.clearEvents()
        keys = self.presentStim(stim, trial['stimulusDuration'], mode = 'task')
        trialClock.reset()
        print(keys)
        for key,response_time in keys:
            choice = self.action_keys.index(key)
            print('Choice: %s' % choice)
            trial['response'] = choice
            trial['rt'] = trialClock.getTime()
            # get feedback
            correct_choice = self.getCorrectChoice(stim,trial['ts'])
            if correct_choice == choice:
                FB = trial['reward_amount']
            else:
                FB = trial['punishment_amount']
             #record points for bonus
            self.pointtracker += FB
            #If training, present FB to window
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
        #If subject did not respond within the stimulus window clear the stim
        #and admonish the subject
        if trial['rt']==999:
            self.clearWindow()            
            core.wait(trial['FBonset'])
            self.presentTextToWindow('Please Respond Faster')
            core.wait(trial['FBDuration'])
            self.clearWindow()
        
        # log trial and add to data
        self.writeToLog(json.dumps(trial))
        self.alldata.append(trial)
        return trial
            
        

    def run_task(self, pause_trial = None):
        self.startTime = core.getTime()
        pause_time = 0
        for trial in self.stimulusInfo:
            if trial == pause_trial:
                time1 = core.getTime()
                self.presentTextToWindow("Take a break! Press '5' when you're ready to continue.")
                self.waitForKeypress(self.trigger_key)
                self.clearWindow()
                pause_time = core.getTime() - time1
            
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
        self.presentTextToWindow('Thank you. Please wait for the experimenter.')
        self.waitForKeypress(self.quit_key)
        self.closeWindow()
















