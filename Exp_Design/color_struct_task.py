"""
generic task using psychopy
"""

from psychopy import visual, core, event, logging, data, misc
import sys,os
import yaml
import numpy as np
import datetime
import json
import random as r

try:
    from save_data_to_db import *
except:
    pass

def np_to_list(d):
    d_fixed={}
    for k in d.iterkeys():
        if isinstance(d[k],np.ndarray) and d[k].ndim==1:
            d_fixed[k]=[x for x in d[k]]
            print 'converting %s from np array to list'%k
        else:
            #print 'copying %s'%k
            d_fixed[k]=d[k]
    return d_fixed


class colorStructTask:
    """ class defining a psychological experiment
    """
    
    def __init__(self,config_file,subject_code,verbose=True, 
                 fullscreen = False, bot = None, mode = 'FB'):
            
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
        self.trigger_times=[]
        self.config_file=config_file
        self.trialnum = 0
        self.track_response = []
        self.fullscreen = fullscreen
        self.pointtracker = 0
        self.bot = bot
        #Choose 'practice', 'FB', 'noFB'
        self.mode = mode
        try:
            self.loadStimulusFileYAML(config_file)
        except:
            print 'cannot load config file'
            sys.exit()
                                                        
        self.logfilename='%s_%s_%s.log'%(self.subject_code,self.taskname,self.timestamp)
        self.datafilename='%s_%s_%s'%(self.subject_code,self.taskname,self.timestamp)

    def loadStimulusFileYAML(self,filename):
        """ load a stimulus file in YAML format
        """
        if not os.path.exists(filename):
            raise BaseException('Stimulus file not found')
        yaml_iterator=yaml.load_all(file(filename,'r'))
        for trial in yaml_iterator:
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
         
    def writeData(self, loc = '../Data/'):
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
        self.win.setColor('black')
        self.win.flip()
        self.win.flip()
        
    def presentTextToWindow(self,text, color = u'white'):
        """ present a text message to the screen
        return:  time of completion
        """
        
        if not self.textStim:
            self.textStim=visual.TextStim(self.win, text=text,font='BiauKai',
                                height=1,color=color, colorSpace=u'rgb',
                                opacity=1,depth=0.0,
                                alignHoriz='center',wrapWidth=50)
            self.textStim.setAutoDraw(True) #automatically draw every frame
        else:
            self.textStim.setText(text)
            self.textStim.setColor(color)
        self.win.flip()
        return core.getTime()

    def defineStims(self, stims = None):
        if stims:
            self.stims = stims
        else:
            if self.mode == 'FB':
                self.stims = [visual.ImageStim(self.win, image = '../Stimuli/93.tiff', units = 'cm', size = (7, 7)),
                            visual.ImageStim(self.win, image = '../Stimuli/22.tiff', units = 'cm', size = (7, 7))]
                r.shuffle(self.stims)
            elif self.mode == 'Practice':
                self.stims = [visual.ImageStim(self.win, image = '../Stimuli/12.tiff', units = 'cm', size = (7, 7)),
                            visual.ImageStim(self.win, image = '../Stimuli/17.tiff', units = 'cm', size = (7, 7))]
            

        
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
        
    def getPoints(self):
        return (self.pointtracker,self.trialnum)
        
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
        self.win.setColor(trial['context'])
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
                choice = self.bot.choose(trial['stim'])
                core.wait(choice[1])
                key_response = [(choice[0], core.getAbsTime())]
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
                        #trial['stimulusCleared']=core.getTime()-onsetTime
                        trial['stimulusCleared']=trialClock.getTime()
                        core.wait(trial['FBonset'])    
                        #If training, present FB
                        if self.mode != "noFB":
                            trial['actualFBOnsetTime'] = trialClock.getTime()-trial['stimulusCleared']
                            choice = self.action_keys.index(key)
                            FB = trial['state_content']['ts'][stim_i,choice]
                            if self.bot:
                                self.bot.learn(FB)
                                print(self.bot.Q.Qstates)
                            trial['FB'] = FB
                            if FB == 1:
                                self.presentTextToWindow('+1 point')
                            else:
                                self.presentTextToWindow('+' + FB + ' points')
                            core.wait(self.FBDuration)
                            self.clearWindow()     
        #If subject did not respond within the stimulus window clear the stim
        #and admonish the subject
        if trial['stimulusCleared']==0:
            self.clearWindow()
            trial['stimulusCleared']=trialClock.getTime()
            trial['response'].append('NA')
            trial['rt'].append(999)
            core.wait(.5)
            self.presentTextToWindow('Please Respond Faster')
            core.wait(1)
            self.clearWindow()
        return trial
            
        



















