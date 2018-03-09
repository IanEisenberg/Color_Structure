"""
generic task using psychopy
"""
from BaseExp import BaseExp
import json
import numpy as np
from psychopy import visual, core, event
import subprocess
import sys,os
import yaml
from flowstim import OpticFlow

class probContextTask(BaseExp):
    """ class defining a probabilistic context task
    """
    
    def __init__(self,config_file,subjid,save_dir,verbose=True, 
                 cue_type='probabilistic', win_kwargs={}):
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
            print('cannot load config file')
            sys.exit()
            
        self.aperture=None
        
        self.trialnum = 0
        self.track_response = []
        self.pointtracker = 0
        self.cue_type = cue_type
        # init Base Exp
        super(probContextTask, self).__init__(self.taskname, subjid, save_dir, 
                                              win_kwargs)
    
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
    def defineStims(self, stim = None, cue = None):
        ratio = self.win.size[1]/float(self.win.size[0])
        stim_height = 1
        ratio = .3
        cue_height = .05
        if stim == None:
            self.stim=OpticFlow(self.win, 
                                speed=self.base_speed,
                                color=[1,1,1], 
                                nElements = 2000,
                                sizes=[stim_height*ratio, stim_height])
        else:
            self.stim = stim
        if cue == None:
            # set up cue
            self.cue = visual.Circle(self.win,units = 'norm',
                                     radius = (cue_height*ratio*5, cue_height),
                                     fillColor = 'white', edges = 120)
        else:
            self.cue = cue
    
    def presentCue(self, trial, duration):
        print('cue', trial['ts'], self.cue_type)
        if self.cue_type == 'deterministic':
            self.text_stim.setText(trial['ts'])
            self.text_stim.setHeight(.15)
            self.text_stim.setColor(self.text_color)
            self.text_stim.draw()
        elif self.cue_type == 'probabilistic':  
            self.cue.setPos((0, trial['context']*.8))
            self.cue.draw()
        self.win.flip()
        core.wait(duration)
        self.win.flip()
        
    def presentTrial(self,trial):
        """
        This function presents a stimuli, waits for a response, tracks the
        response and RT and presents appropriate feedback. This function also controls the timing of FB 
        presentation.
        """
        trialClock = core.Clock()
        self.trialnum += 1
        trial['actualOnsetTime']=core.getTime() - self.startTime
        trial['stimulusCleared']=0
        trial['response'] = np.nan
        trial['rt'] = np.nan
        trial['FB'] = np.nan
        
        
        stim = trial['stim']
        trial_attributes = self.getTrialAttributes(stim)
        
        print('*'*40)        
        print('Taskset: %s\nSpeed: %s, Strength: %s \
              \nOriDirection: %s, OriStrength: %s \
              \nCorrectChoice: %s' % 
              (trial['ts'], 
               stim['speedDirection'], stim['speedStrength'], 
               stim['oriDirection'],stim['oriStrength'],
               self.getCorrectChoice(trial_attributes,trial['ts'])))

        
        # present cue
        self.presentCue(trial, trial['cueDuration'])
        core.wait(trial['CSI'])
        # present stimulus and get response
        event.clearEvents()
        trialClock.reset()
        key_response = self.presentStim(trial_attributes, 
                                        duration=trial['stimulusDuration'], 
                                        response_window=trial['responseWindow'], 
                                        SRI=trial['stimResponseInterval'],
                                        clock=trialClock)
        if key_response:
            # record response
            trial['response'], trial['rt'] = key_response
            print('Choice: %s' % trial['response'])
            # get feedback and update tracker
            correct_choice = self.getCorrectChoice(trial_attributes,trial['ts'])
            #update tracker if in trial mode
            if correct_choice == trial['response']:
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
        else:
            self.clearWindow()            
            core.wait(trial['FBonset'])
            self.presentTextToWindow('Please Respond Faster')
            core.wait(trial['FBDuration'])
            self.clearWindow()
        
        # log trial and add to data
        self.writeToLog(json.dumps(trial))
        self.alldata.append(trial)
        return trial
            
    def run_task(self, intro_text=None):
        self.setupWindow()
        self.defineStims()
         # set up pause trials
        length_min = self.stimulusInfo[-1]['onset']/60
        # have break every 6 minutes
        num_pauses = np.round(length_min/6)
        pause_trials = np.round(np.linspace(0,self.exp_len,num_pauses+1))[1:-1]
        pause_time = 0
        timer_text = "Take a break!\n\nContinue in: \n\n       "
        # present intro screen
        if intro_text:
            self.presentInstruction(intro_text)
        # get ready
        self.presentTextToWindow('Get Ready!', size=.15)
        core.wait(1.5)
        # start the task
        self.startTime = core.getTime()
        self.clearWindow(fixation=self.fixation)
        for trial in self.stimulusInfo:
            if trial['trial_count'] in pause_trials:
                time1 = core.getTime()
                self.presentTimer(duration=30, text=timer_text)
                self.clearWindow()
                pause_time += core.getTime() - time1
            
            # wait for onset time
            while core.getTime() < trial['onset'] + self.startTime + pause_time:
                    key_response=event.getKeys([self.quit_key])
                    if len(key_response)==1:
                        self.shutDownEarly()
            self.presentTrial(trial)
        
        # clean up and save
        other_data={'taskinfo': self.taskinfo,
                    'configfile': self.config_file}
        self.writeData(taskdata=self.alldata,
                       other_data=other_data)
        self.presentInstruction(
            """
            Thank you. Please wait for the experimenter.
            """)
        self.closeWindow()
















