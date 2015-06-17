"""
generic task using psychopy
"""

from psychopy import visual, core, event
import sys,os
import yaml
import numpy as np
import datetime
import json
import subprocess


class probContextTask:
    """ class defining a probabilistic context task
    """
    
    def __init__(self,config_file,subject_code,verbose=True, 
                 fullscreen = False, mode = 'task'):
            
        self.subject_code=subject_code
        self.win=[]
        self.window_dims=[800,600]
        self.textStim=[]
        self.stims=[]
        self.stimulusInfo=[]
        self.loadedStimulusFile=[]
        self.startTime=[]
        self.alldata=[]
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        #looks up the hash of the most recent git push. Stored in log file
        self.gitHash = subprocess.check_output(['git','rev-parse','--short','HEAD'])[:-1]
        self.trigger_times=[]
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
            self.loadStimulusFileNP(config_file)
        except:
            print(mode + ': cannot load config file')
            sys.exit()
                                                        
        self.logfilename='%s_%s_%s.log'%(self.subject_code,self.taskname,self.timestamp)
        self.datafilename='%s_%s_%s'%(self.subject_code,self.taskname,self.timestamp)

    def loadStimulusFileNP(self,filename):
        """ load a stimulus file in numpy format
        """
        if not os.path.exists(filename):
            raise BaseException('Stimulus file not found')
        config_file = np.load(filename)
        for trial in config_file:
            if trial.has_key('taskname'):
                self.taskinfo=trial
                for k in self.taskinfo.iterkeys():
                    self.__dict__[k]=self.taskinfo[k]
            else:
                self.stimulusInfo.append(trial)
        if len(self.stimulusInfo)>0:
            self.loadedStimulusFile=filename
            
    def toJSON(self):
        """ log the initial conditions for the task. Exclude the list of all
        trials (stimulusinfo), the bot, and taskinfo (self.__dict__ includes 
        all of the same information as taskinfo)
        """
        init_dict = {k:self.__dict__[k] for k in self.__dict__.iterkeys() if k 
                    not in ('clock', 'stimulusInfo', 'alldata', 'bot', 'taskinfo')}
        return json.dumps(init_dict)
    
    def writeToLog(self,msg,loc = '../Log/'):
        f=open(str(loc) + self.logfilename,'a')
        f.write(msg)
        f.write('\n')
        f.close()
         
    def writeData(self, loc = '../RawData/'):
        data = {}
        data['taskinfo']=self.taskinfo
        data['configfile']=self.config_file
        data['subcode']=self.subject_code
        data['timestamp']=self.timestamp
        data['taskdata']=self.alldata
        f=open(str(loc) + self.datafilename + '.yaml','w')
        yaml.dump(data,f)
    
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

    def defineStims(self, stims = None):
        """Defines two sets of stims for practice and the task.
            Ratio variable is used due to the normalized units to make 
            'square' squares
        """
        if stims:
            self.stims = stims
        else:
            height = .2
            ratio = self.win.size[1]/float(self.win.size[0])
            if self.mode == 'practice':
                self.stims = {self.stim_ids[0]: visual.Polygon(self.win,units = 'norm',radius = (height*ratio/2, height/2),edges = 3,fillColor = 'green'),
                              self.stim_ids[1]: visual.Polygon(self.win, units = 'norm',radius = (height*ratio/2, height/2),edges = 3,fillColor = 'yellow'),
                              self.stim_ids[2]: visual.Polygon(self.win, units = 'norm',radius = (height*ratio/2, height/2),edges = 5,fillColor = 'green'),
                              self.stim_ids[3]: visual.Polygon(self.win, units = 'norm',radius = (height*ratio/2, height/2),edges = 5,fillColor = 'yellow')}

            elif self.mode == 'task':
                self.stims = {self.stim_ids[0]: visual.Rect(self.win,height*ratio, height,units = 'norm', fillColor = 'red'),
                              self.stim_ids[1]: visual.Rect(self.win,height*ratio, height,units = 'norm',fillColor = 'blue'),
                              self.stim_ids[2]: visual.Circle(self.win,units = 'norm',radius = (height*ratio/2, height/2),edges = 32,fillColor = 'red'),
                              self.stim_ids[3]: visual.Circle(self.win,units = 'norm',radius = (height*ratio/2, height/2), edges = 32,fillColor = 'blue')}
        
            
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

    def waitSeconds(self,duration):
        """ wait for some amount of time (in seconds)
        """
        
        core.wait(duration)
        
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
        
    def presentStims(self, mode = 'practice'):
        """ Used during instructions to present possible stims
        """
        height = .2
        ratio = self.win.size[1]/float(self.win.size[0])
        pos = [-.6, -.2, .2, .6]
        if mode == 'practice':
                stims = [visual.Polygon(self.win,units = 'norm',radius = (height*ratio/2, height/2),edges = 3,fillColor = 'green'),
                         visual.Polygon(self.win,units = 'norm',radius = (height*ratio/2, height/2),edges = 3,fillColor = 'yellow'),
                         visual.Polygon(self.win, units = 'norm',radius = (height*ratio/2, height/2),edges = 5,fillColor = 'green'),
                         visual.Polygon(self.win, units = 'norm',radius = (height*ratio/2, height/2),edges = 5,fillColor = 'yellow')]

        elif mode == 'task':
            stims = [visual.Rect(self.win,height*ratio, height,units = 'norm', fillColor = 'red'),
                     visual.Rect(self.win,height*ratio, height,units = 'norm',fillColor = 'blue'),
                     visual.Circle(self.win,units = 'norm',radius = (height*ratio/2, height/2),edges = 32,fillColor = 'red'),
                     visual.Circle(self.win,units = 'norm',radius = (height*ratio/2, height/2), edges = 32,fillColor = 'blue')]
        
        for i,stim in enumerate(stims):
            stim.setPos((pos[i],0))
            stim.draw()
        self.win.flip()

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
        stim_i = trial['stim']
        if self.botMode != 'short':
            self.stims[stim_i].setPos((0, trial['context']*.8))
            self.stims[stim_i].draw()
            self.win.flip()
        trialClock.reset()
        event.clearEvents()
        trial['actualOnsetTime']=core.getTime() - self.startTime
        trial['stimulusCleared']=0
        trial['response']=[]
        trial['rt']=[]
        trial['FB'] = []
        while trialClock.getTime() < (self.stimulusDuration):
            key_response=event.getKeys(None,True)
            if self.bot:
                bot_action = self.bot.choose(trial['stim'], trial['context'])
                choice = self.action_keys.index(bot_action[0])
                trial['response'].append(choice)
                if choice == stim_i[trial['ts']]:
                    FB = trial['reward']
                else:
                    FB = trial['punishment']
                if trial['FBDuration'] != 0:
                    trial['FB'] = FB
                if self.botMode == 'short':
                    trial['stimulusCleared']=trialClock.getTime()
                    trial['actualFBOnsetTime'] = trialClock.getTime()
                    trial['rt'].append(bot_action[1])
                    self.pointtracker += FB
                    break
                else:
                    key_response = [(bot_action[0], bot_action[1])]
                    core.wait(bot_action[1])
            if len(key_response)==0:
                continue
            for key,response_time in key_response:
                if self.quit_key==key:
                    self.shutDownEarly()
                elif self.trigger_key==key:
                    self.trigger_times.append(response_time-self.startTime)
                    continue
                elif key in self.action_keys:
                    trial['response'].append(key)
                    trial['rt'].append(trialClock.getTime())
                    if self.clearAfterResponse and trial['stimulusCleared']==0:
                        self.clearWindow()
                        trial['stimulusCleared']=trialClock.getTime()
                        choice = self.action_keys.index(key)
                        if choice == stim_i[trial['ts']]:
                            FB = trial['reward']
                        else:
                            FB = trial['punishment']
                        #record points for bonus
                        self.pointtracker += FB
                        #If training, present FB to window
                        if trial['FBDuration'] != 0:
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
            core.wait(.5)
            self.presentTextToWindow('Please Respond Faster')
            core.wait(1)
            self.clearWindow()
        return trial
            
        



















