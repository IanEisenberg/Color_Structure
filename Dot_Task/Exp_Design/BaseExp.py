import pickle
import datetime
import json
import numpy as np
from psychopy import core, event, visual
import os
import sys
from Dot_Task.Exp_Design.flowstim import Fixation
import yaml
        
class BaseExp(object):
    """ class defining a probabilistic context task
    """
    
    def __init__(self, expid, subjid, save_dir, win_kwargs={}):
        self.expid = expid
        self.subjid=subjid
        self.save_dir = save_dir  
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # set up static variables
        self.quit_key = 'q'
        self.trigger_key = '5'
        self.text_color = [1]*3
        self.text_stim = None
        self.fixation = None
        # set up window
        self.win=[]
        if 'size' not in win_kwargs.keys():
            win_kwargs['size'] = [1920,1200]
        self.win_kwargs = win_kwargs
        
        # set up recording files
        self.logfilename='%s_%s_%s.log'%(self.subjid,self.expid,self.timestamp)
        self.datafilename='%s_%s_%s.pkl'%(self.subjid,self.expid,self.timestamp)
        
    def loadConfigFile(self,filename):
        """ load a config file from yaml
        """
        if not os.path.exists(filename):
            raise BaseException('Config file not found')
        config_file = yaml.load(open(filename,'r'))
        for trial in config_file:
            if 'taskname' in trial.keys():
                self.taskinfo=trial
                for k in self.taskinfo.keys():
                    self.__dict__[k]=self.taskinfo[k]
            else:
                self.stimulusInfo.append(trial)
        if len(self.stimulusInfo)>0:
            self.loadedStimulusFile=filename

    #**************************************************************************
    # ******* Function to Save Data **************
    #**************************************************************************
    
    def toJSON(self, excluded_keys=[]):
        """ log the initial conditions for the task."""
        init_dict = {k:self.__dict__[k] for k in self.__dict__.keys() if k 
                    not in excluded_keys}
        return json.dumps(init_dict)
    
    def writeToLog(self,msg):
        log_loc = os.path.join(self.save_dir,'Log',self.subjid,self.logfilename)
        try:
            os.makedirs(os.path.dirname(log_loc))
        except OSError:
            pass
        f = open(log_loc,'a')
        f.write(msg)
        f.write('\n')
        f.close()
         
    def writeData(self, taskdata={}, other_data={}):
        # save data
        save_loc = os.path.join(self.save_dir,'RawData',self.subjid,self.datafilename)
        try:
            os.makedirs(os.path.dirname(save_loc))
        except OSError:
            pass
        data = {}
        data['subjid']=self.subjid
        data['taskdata'] = taskdata
        data['timestamp']=self.timestamp
        data.update(other_data)
        try:
            f=open(save_loc,'wb')
        except IOError:
            os.makedirs(os.path.split(save_loc)[0])
            f=open(save_loc,'wb')
        pickle.dump(data,f)
    
    def checkRespForQuitKey(self,resp):
            if self.quit_key in resp:
                self.shutDownEarly()
    
    def clearWindow(self, fixation=True, fixation_color=None):
        """ clear the main window
        """
        if fixation and self.fixation is not None:
            self.fixation.draw(color=fixation_color)
        if self.text_stim:
            self.text_stim.setText('')
        else:
            self.presentTextToWindow('', flip=False)
        self.win.flip()
            
    def closeWindow(self):
        """ close the main window
        """
        if self.win:
            self.win.close()
        
    def getActions(self):
        return self.action_keys
    
    def getPastAcc(self, time_win):
        """Returns the ratio of hits/trials in a predefined window
        """
        if time_win > self.trialnum:
            time_win = self.trialnum
        return sum(self.track_response[-time_win:])
    
    def getPoints(self):
        return (self.pointtracker,self.trialnum)
    
    def getSquareSize(self, win, size=.3):
            stim_ratio = float(win.size[0])/win.size[1]
            square_size = np.array([size, stim_ratio*size])
            return square_size
        
    def getStims(self):
        return self.stims
    
    def getTSorder(self):
        return [self.taskinfo['states'][0]['ts'],
                self.taskinfo['states'][1]['ts']]
        
    
    def presentStim(self, trial_attributes, duration=.5, response_window=3,
                    SRI=0):
        """ Used during instructions to present possible stims
        """
        # unpack trial attributes
        ss = trial_attributes['speed_start']
        se = trial_attributes['speed_end']
        os = trial_attributes['ori_start']
        oe = trial_attributes['ori_end']
        md = trial_attributes['motion_direction']
        # reset dot position
        self.stim.setupDots()
        self.stim.updateTrialAttributes(dir=md,ori=os,speed=ss)            
        if self.aperture: self.aperture.enable()
        # display stimulus
        stim_clock = core.Clock()
        while stim_clock.getTime() < duration:
            percent_complete = stim_clock.getTime()/duration
            # smoothly move color over the duration
            orientation = os*(1-percent_complete) + oe*percent_complete
            # change speed
            speed = ss*(1-percent_complete) + se*percent_complete
            self.stim.updateTrialAttributes(ori=orientation, speed=speed)
            self.stim.draw()
            keys = event.getKeys([self.quit_key])
            self.checkRespForQuitKey(keys)
        # wait stim-respones interval
        if SRI>0: 
            self.clearWindow(fixation=True)
            core.wait(SRI)
        # indicate response window and wait for response
        self.clearWindow(fixation=True, fixation_color='#0099ff')
        stim_clock.reset()
        key_response = event.waitKeys(response_window,
                                      self.action_keys + [self.quit_key],
                                      timeStamped=stim_clock)
        if key_response is not None:
            assert len(key_response) == 1
            key_response = key_response[0]
            self.checkRespForQuitKey([key_response[0]])
        else:
            key_response = []
        if self.aperture: self.aperture.disable()
        return key_response
            
    def getCorrectChoice(self,trial_attributes,ts):
        # action keys are set up as the choices for ts1 followed by ts2
        # so the index for the correct choice must take that into account
        if ts == 'motion':
            correct_choice = int(trial_attributes['speed_end']>trial_attributes['speed_start'])
        elif ts == 'orientation':
            # correct choice is based on whether the orientation become more or less positive
            correct_choice = int(trial_attributes['ori_end']>trial_attributes['ori_start'])+2
        return self.action_keys[correct_choice]
    
    def getTrialAttributes(self,stim):
        ss, sd, os, od, md, oriBase = [stim[k] for k in 
                                     ['speedStrength','speedDirection',
                                      'oriStrength','oriDirection',
                                      'motionDirection', 'oriBase']]
        # transform word difficulties into numbers
        ss = self.speed_difficulties[(md, ss)]
        os = self.ori_difficulties[(oriBase, os)]
        # create start and end points
        speed_start = self.base_speed
        speed_end = self.base_speed + ss*sd
        
        ori_start = oriBase
        ori_end = oriBase + os*od

        return {'speed_start': speed_start, 
                'speed_end': speed_end, 
                'ori_start': ori_start, 
                'ori_end': ori_end, 
                'motion_direction': md}
    
    def presentInstruction(self, text, keys=None, size=.07):
            if keys is None:
                keys = self.trigger_key
            self.presentTextToWindow(text, size=size)
            resp,time=self.waitForKeypress(keys)
            self.checkRespForQuitKey(resp)
            event.clearEvents()
            return resp[0][0]
            
    def presentTextToWindow(self, text, size=.15, color=None, duration=None,
                            position=None, flip=True, fixation=False):
        """ present a text message to the screen
        return:  time of completion
        """
        if color is None:
            color = self.text_color
        if position is None:
            position = (0,0)
        if not self.text_stim:
            self.text_stim=visual.TextStim(self.win, 
                                          text=text,
                                          font='BiauKai',
                                          pos=position,
                                          height=size,
                                          color=color, 
                                          colorSpace=u'rgb',
                                          opacity=1,
                                          depth=0.0,
                                          alignHoriz='center',
                                          alignVert='center', 
                                          wrapWidth=50)
        else:
            self.text_stim.setText(text)
            self.text_stim.setHeight(size)
            self.text_stim.setColor(color)
            self.text_stim.pos = position
        self.text_stim.draw()
        if fixation:
            fixation.draw()
        if flip:
            self.win.flip()
        if duration:
            core.wait(duration)
        return core.getTime()
    
    def presentTimer(self, duration, timer_position=None, text=None,
                     countdown=True):
        """ Presents a timer to the subject
        
        Args:
            Duration: integer. How many seconds should the timer last?
            timer_position: tuple, passed to visual.TextStim
            text: optional text to embed the time in. Time will be added to end
                of text
        """
        clock = core.Clock()
        while clock.getTime() < duration:
            time = int(clock.getTime())
            if countdown:
                time = duration-time
            if text:
                timer_text = text + '{0: ^5}'.format(time)
            else:
                timer_text = time
            self.presentTextToWindow(timer_text, position=timer_position)
        self.win.flip()
        
    def setupWindow(self, aperture=True):
        """ set up the main window
        """
        self.win = visual.Window(monitor='testMonitor', 
                                 units='norm', 
                                 allowStencil=True,
                                 color=[-1,-1,-1],
                                 **self.win_kwargs)   
        if aperture==True:
            # define aperture
            aperture_size = 1.5
            aperture_vertices = visual.Aperture(self.win, size=aperture_size, units='norm').vertices
            ratio = float(self.win.size[1])/self.win.size[0]
            aperture_vertices[:,0]*=ratio
            self.aperture = visual.Aperture(self.win, size=aperture_size, units='norm', shape = aperture_vertices)
            self.aperture.disable()                     
        self.win.flip()
        
    def shutDownEarly(self):
        self.closeWindow()
        sys.exit()
            
    def waitForKeypress(self,keyList=[], clear=True):
        """ wait for a keypress and return the pressed key
        - this is primarily for waiting to start a task
        - use getResponse to get responses on a task
        """
        if type(keyList) == str:
            keyList = [keyList]
        if len(keyList) == 0:
            keyList = [self.trigger_key]
        keyList.append(self.quit_key)
        start=False
        event.clearEvents()
        while start==False:
            keys = event.getKeys(keyList=keyList,
                                 timeStamped=True)
            for k,response_time in keys:
                start = True
                self.checkRespForQuitKey(k)
        if clear==True:
            self.clearWindow()
        return keys, core.getTime()
        
    def runTask(self):
        self.setupWindow()
        self.stim_size = self.getSquareSize(self.win)
        self.presentInstruction('Welcome! Press 5 to continue...')
        self.waitForKeypress()
        
