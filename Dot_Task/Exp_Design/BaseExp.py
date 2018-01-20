import cPickle
import datetime
import json
import numpy as np
from psychopy import core, event, visual
import os
import sys


        
class BaseExp(object):
    """ class defining a probabilistic context task
    """
    
    def __init__(self, expid, subjid, save_dir, fullscreen = False):
        self.expid = expid
        self.subjid=subjid
        self.save_dir = save_dir  
        self.fullscreen = fullscreen
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # set up static variables
        self.quit_key = 'q'
        self.trigger_key = '5'
        self.text_color = [1]*3
        self.text_stim = None
        # set up window
        self.win=[]
        self.window_dims=[1920,1080]
        
        # set up recording files
        self.logfilename='%s_%s_%s.log'%(self.subjid,self.expid,self.timestamp)
        self.datafilename='%s_%s_%s.pkl'%(self.subjid,self.expid,self.timestamp)
            
    #**************************************************************************
    # ******* Function to Save Data **************
    #**************************************************************************
    
    def toJSON(self, excluded_keys=[]):
        """ log the initial conditions for the task."""
        init_dict = {k:self.__dict__[k] for k in self.__dict__.iterkeys() if k 
                    not in excluded_keys}
        return json.dumps(init_dict)
    
    def writeToLog(self,msg):
        f = open(os.path.join(self.save_dir,'Log',self.logfilename),'a')
        f.write(msg)
        f.write('\n')
        f.close()
         
    def writeData(self, taskdata={}, other_data={}):
        # save data
        save_loc = os.path.join(self.save_dir,'RawData',self.datafilename)
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
        cPickle.dump(data,f)
    
    def checkRespForQuitKey(self,resp):
            if self.quit_key in resp:
                self.shutDownEarly()
    
    def clearWindow(self, fixation=None):
        """ clear the main window
        """
        if fixation:
            fixation.draw()
        if self.text_stim:
            self.text_stim.setText('')
            self.win.flip()
        else:
            self.presentTextToWindow('')
            
    def closeWindow(self):
        """ close the main window
        """
        if self.win:
            self.win.close()
            
    def getSquareSize(self, win, size=.3):
            stim_ratio = float(win.size[0])/win.size[1]
            square_size = np.array([size, stim_ratio*size])
            return square_size
                
    def presentInstruction(self, text, size=.07):
            self.presentTextToWindow(text, size=size)
            resp,time=self.waitForKeypress(self.trigger_key)
            self.checkRespForQuitKey(resp)
            event.clearEvents()
    
    def presentTextToWindow(self, text, size=.15, color=None, duration=None,
                            position=None, flip=True):
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
            self.text_stim.setColor(self.text_color)
        self.text_stim.draw()
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
        
    def setupWindow(self, **kwargs):
            """ set up the main window
            """
            additional_kwargs = {'screen': 1, 
                                 'monitor': 'testMonitor'}
            additional_kwargs.update(kwargs)
            self.win = visual.Window(self.window_dims, allowGUI=False, 
                                     fullscr=self.fullscreen, units='norm',
                                     allowStencil=True, color=[-1,-1,-1], 
                                     **additional_kwargs)   
    
            self.win.flip()
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
        
