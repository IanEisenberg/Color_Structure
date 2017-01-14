"""
generic task using psychopy
"""

from psychopy import visual, core, event
import sys,os
import json
import yaml
import datetime
import subprocess
from dot_stim import getDotStim

class probContextTask:
    """ class defining a probabilistic context task
    """
    
    def __init__(self,config_file,subjid,save_dir,verbose=True, 
                 fullscreen = False, color_coherences = None, 
                 motion_coherences = None, mode = 'task'):
                     
        self.save_dir = save_dir  
        self.subjid=subjid
        # set up window
        self.win=[]
        self.window_dims=[800,600]
        # set up stim
        if color_coherences == None:
            self.color_coherences = {'easy': .7, 'medium': .4, 'hard': .1}
        else:
            self.color_coherences = color_coherences
        if motion_coherences == None:
            self.motion_coherences = {'easy': .7, 'medium': .4, 'hard': .1}
        else:
            self.motion_coherences = motion_coherences
        self.tasksets = ['motionDirection','colorMajority']
        self.textStim=[]
        self.stimulusInfo=[]
        self.loadedStimulusFile=[]
        self.startTime=[]
        self.alldata=[]
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        #looks up the hash of the most recent git push. Stored in log file
        self.gitHash = subprocess.check_output(['git','rev-parse','--short','HEAD'])[:-1]
        self.config_file=config_file
        self.trialnum = 0
        self.track_response = []
        self.fullscreen = fullscreen
        self.text_color = [1]*3
        self.pointtracker = 0
        #intializes a bot variable that can be set by setBot
        self.bot = None
        self.botMode = None
        #Choose 'practice', 'task': determines stimulus set to use
        self.mode = mode
        try:
            self.loadConfigFile(config_file)
            self.stim=getDotStim(color_coherence = color_coherences['medium'],
                             motion_coherence = motion_coherences['medium'],
                             direction = self.stim_motions[0],
                             colors = self.stim_colors)
        except:
            print(mode + ': cannot load config file')
            sys.exit()
                                                        
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
                    not in ('clock', 'stimulusInfo', 'alldata', 'bot', 'taskinfo')}
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
        self.win = visual.Window(self.window_dims, allowGUI=False, fullscr=self.fullscreen, 
                                 monitor='testMonitor', units='deg')                        
        self.win.setColor([-1,-1,-1],'rgb')
        self.win.flip()
        self.win.flip()
        
    def presentTextToWindow(self,text):
        """ present a text message to the screen
        return:  time of completion
        """
        
        if not self.textStim:
            self.textStim=visual.TextStim(self.win, text=text,font='BiauKai',
                                height=1,color=self.text_color, colorSpace=u'rgb',
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
        
    def setBot(self, bot, mode = "full"):
        """ sets up a bot to run the experiment
            mode = 'full' displays the experiment like normal.
            mode = 'short' doesn't display any images and just create data
        """
        self.bot = bot
        self.botMode = mode
    
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
        
    def presentContexts(self):
        """ Used during instructions to present possible stimulus positions
        """
        height = .05
        ratio = self.win.size[1]/float(self.win.size[0])
        tmp_stim = visual.Rect(self.win,height*ratio*5, height,units = 'norm',fillColor = 'yellow')
        for context in self.context_means:
            tmp_stim.setPos((0, context*.9))
            tmp_stim.draw()
        self.win.flip()
        
    def presentStims(self, stim, duration = .5, mode = 'practice'):
        """ Used during instructions to present possible stims
        """
        ms,md,cs,ci = [stim[k] for k in ['motionStrength','motionDirection','colorStrength','colorIdentity']]
        # convert two dimensions of color (Strength and direction) to one dimension, percentage of the first color
        cc = abs(self.colorCoherences[cs]-ci)
        if mode == 'practice':
            self.stim.setColorCoherence(cc)
            self.stim.Coherence(self.motionCoherences['easy'])
            self.stim.setDir(stim.motions[stim.md])
        elif mode == 'task':
            self.stim.setColorCoherence(self.colorCoherences[cs])
            self.stim.Coherence(self.motionCoherences[ms])
        stim_clock = core.Clock()
        while stim_clock.getTime() < duration:
            self.stim.draw()
            self.win.flip()
            keys = event.getKeys(None,True)
            if len(keys) > 0:
                break
        return keys
            
    def getCorrectChoice(stim,ts):
        # action keys are set up as the choices for ts1 followed by ts2
        # so the index for the correct choice must take that into account
        correct_choice = stim[self.tasksets[ts]] + self.tasksets[ts]*2
        return correct_choice
        
    def presentTrial(self,trial):
        """
        This function presents a stimuli, waits for a response, tracks the
        response and RT and presents appropriate feedback. If a bot was loaded
        the bot interacts with the experiment through this function by supplying
        'actions' and 'RTs'. This function also controls the timing of FB 
        presentation.
        """
        trialClock = core.Clock()
        self.trialnum += 1
        stim = trial['stim']
        
        event.clearEvents()
        trial['actualOnsetTime']=core.getTime() - self.startTime
        trial['stimulusCleared']=0
        trial['response']=[]
        trial['rt']=[]
        trial['FB'] = []
        # present stimulus and get response
        keys = self.presentStims(stim, trial['stimulusDuration'], mod = 'test')
        trialClock.reset()
        # check for quit key
        if self.quit_key in keys:
            self.shutDownEarly()
        choice = self.action_keys.index(keys[0])
        trial['response'].append(choice)
        # get feedback
        correct_choice = getCorrectChoice(stim,trial['ts'])
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
        if trial['stimulusCleared']==0:
            self.clearWindow()
            trial['stimulusCleared']=trialClock.getTime()
            trial['response'].append(999)
            trial['rt'].append(999)
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
            if not self.bot:
                if trial == pause_trial:
                    time1 = core.getTime()
                    self.presentTextToWindow("Take a break! Press '5' when you're ready to continue.")
                    self.waitForKeypress(self.trigger_key)
                    self.clearWindow()
                    pause_time = core.getTime() - time1
            
            #if botMode = short, don't wait for onset times
            if self.botMode != 'short':
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
        if self.bot == None:
            self.presentTextToWindow('Thank you. Please wait for the experimenter.')
            self.waitForKeypress(self.quit_key)
        self.closeWindow()
















