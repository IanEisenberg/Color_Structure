"""
generic task using psychopy
"""
import json
import numpy as np
from psychopy import visual, core, event
import subprocess
import sys
from Exp_Design.BaseExp import BaseExp
from Exp_Design.flowstim import OpticFlow

class probContextTask(BaseExp):
    """ class defining a probabilistic context task
    """
    
    def __init__(self,config_file,subjid,save_dir,fmri_trigger=None,
                 verbose=True, cue_type='probabilistic', win_kwargs={}):
        # set up some variables
        self.stimulusInfo=[]
        self.loadedStimulusFile=[]
        self.expClock = core.Clock()
        self.alldata=[]
        self.fmri_trigger=fmri_trigger
        #looks up the hash of the most recent git push. Stored in log file
        self.gitHash = subprocess.check_output(['git','rev-parse','--short','HEAD'])[:-1]

        try:
            self.loadConfigFile(config_file)
        except:
            print('cannot load config file')
            sys.exit()
            
        self.aperture=None
        
        self.track_response = []
        self.pointtracker = 0
        self.cue_type = cue_type
        # init Base Exp
        super(probContextTask, self).__init__(self.taskname, subjid, save_dir, 
                                              win_kwargs)
            
    #**************************************************************************
    # ******* Function to Save Data **************
    #**************************************************************************
    
    def toJSON(self):
        """ log the initial conditions for the task. Exclude the list of all
        trials (stimulusinfo), the bot, and taskinfo (self.__dict__ includes 
        all of the same information as taskinfo)
        """
        init_dict = {k:self.__dict__[k] for k in self.__dict__.keys() if k 
                    not in ('clock', 'stimulusInfo', 'alldata', 'bot', 'taskinfo','win')}
        return json.dumps(init_dict)
  
    #**************************************************************************
    # ******* Display Functions **************
    #**************************************************************************
    def defineStims(self, stim = None, cue = None):
        ratio = self.win.size[1]/float(self.win.size[0])
        stim_height = .035
        ratio = .3
        cue_height = .05
        if stim == None:
            self.stim=OpticFlow(self.win, 
                                speed=self.base_speed,
                                color=[1,1,1], 
                                nElements = 2500,
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
        # define fixation
        self.fixation = self.stim.fixation
    
    def presentCue(self, trial, duration):
        print('cue', trial['ts'], self.cue_type)
        if self.cue_type == 'deterministic':
            cue = 'S' if trial['ts'] == 'motion' else 'R'
            self.presentTextToWindow(cue, 
                                     size=.13,
                                     color=self.text_color,
                                     position=[0,0],
                                     fixation=None,
                                     flip=False)
        elif self.cue_type == 'probabilistic':  
            self.cue.setPos((0, trial['context']*.8))
            self.cue.draw()
        self.win.flip()
        core.wait(duration)
        self.clearWindow()
    
    def presentPause(self):
        pauseClock = core.Clock()
        timer_text = "Take a break!\n\nContinue in: \n\n       "
        self.presentTimer(duration=20, text=timer_text)
        core.wait(1.5)
        self.aperture.enable()
        pause_time = pauseClock.getTime()
        self.alldata.append({'exp_stage': 'pause',
                             'trial_time':  pause_time})
        return pause_time
    
    def presentTrial(self,trial):
        """
        This function presents a stimuli, waits for a response, tracks the
        response and RT and presents appropriate feedback. This function also controls the timing of FB 
        presentation.
        """
        trialClock = core.Clock()
        trial['exp_stage'] = 'practice'
        trial['stimulusCleared']=0
        trial['response'] = np.nan
        trial['rt'] = np.nan
        # update onset to actual onset
        trial['onset'] = self.expClock.getTime()
        # set up stim
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
        key_response = self.presentStim(trial_attributes, 
                                        duration=trial['stimulusDuration'], 
                                        response_window=trial['responseWindow'], 
                                        SRI=trial['stimResponseInterval'])
        if key_response:
            # record response
            trial['response'], trial['rt'] = key_response
            print('Choice: %s' % trial['response'])
            # get feedback and update tracker
            correct_choice = self.getCorrectChoice(trial_attributes,trial['ts'])
            trial['correct_response'] = correct_choice
            trial['correct'] = (correct_choice == trial['response'])
            #If training, present FB to window
            if trial['displayFB'] == True:
                if trial['FBonset'] > 0: 
                    self.clearWindow(fixation=True)
                    core.wait(trial['FBonset'])  
                if trial['correct']:
                    self.clearWindow(fixation=True,
                                     fixation_color=[-.2, 1, -.2])
                else:
                    self.clearWindow(fixation=True,
                                     fixation_color=[1,-.6,-1])
                core.wait(trial['FBDuration'])
        #If subject did not respond within the stimulus window clear the stim
        #and admonish the subject
        else:
            if trial['FBonset'] > 0: 
                self.clearWindow(fixation=True)
                core.wait(trial['FBonset'])  
            self.clearWindow(fixation=True, fixation_color=[1,-.6,-1])
            core.wait(trial['FBDuration'])
        self.clearWindow(fixation=True)
        # log trial and add to data
        trial['trial_time'] = trialClock.getTime()
        self.writeToLog(json.dumps(trial))
        self.alldata.append(trial)
        if self.fmri_trigger:
            core.wait(trial['ITI'])
        return trial
            
    def run_task(self, intro_text=None, ignored_triggers=16):
        self.setupWindow()
        self.defineStims()
         # set up pause trials
        length_min = self.stimulusInfo[-1]['onset']/60
        # have break every 6 minutes
        num_pauses = np.round(length_min/6)
        pause_trials = np.round(np.linspace(0,self.exp_len,num_pauses+1))[1:-1]
        pause_time = 0
        # present intro screen
        if intro_text:
            self.presentInstruction(intro_text)
        if self.fmri_trigger:
            for _ in range(ignored_triggers+1):
                self.waitForKeypress(self.fmri_trigger)
        # get ready
        self.presentTextToWindow('Get Ready!', size=.15)
        core.wait(1.5)
        # start the task
        self.expClock.reset()
        self.clearWindow(fixation=self.fixation)
        for trial in self.stimulusInfo:
            if trial['trial_count'] in pause_trials:
                pause_time += self.presentPause()
            
            # wait for onset time
            if self.fmri_trigger is None:
                while self.expClock.getTime() < trial['onset']+pause_time:
                        key_response=event.getKeys([self.quit_key])
                        if len(key_response)==1:
                            self.shutDownEarly()
            else:
                self.waitForKeypress(self.fmri_trigger)
            self.presentTrial(trial)
        
        # clean up and save
        other_data={'taskinfo': self.taskinfo,
                    'configfile': self.config_file}
        self.writeData(taskdata=self.alldata,
                       other_data=other_data)
        self.presentInstruction(
            """
            End of Run (press 5 to continue)
            """)
        self.closeWindow()
















