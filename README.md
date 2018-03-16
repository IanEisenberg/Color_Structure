# Prob Context Task
## Organization of repo

#### Color\_Shape\_Task: Colored-Shape Stimuli

FYP. Stimuli are easily mapped to responses. The task-set is unclear and must be inferred

#### Dot\_Task: Motion/Orientation Stimuli

Taskset and stimuli are noisy. 

Organization of Task:

* Data
 * Raw Data: Raw behavioral data
 * Log: Log of behavioral data run - only needed if an experiment failed midway through
 * EyeTrackData: Eyetrack data with the same name as the Raw Data
* Config Files
 * Config files are used to determine the structure of each subject's experimental session.  
   The config files is saved here
* Exp_Design
 * Files for different experiment conditions and running the experiment itself
   * run\_adaptive\_procedure.py: get threshold stim level
   * run\_cued\_task: cue subject to different TS (no TS uncertainty)
   * run\_context\_task: (stim and TS uncertainty)
* Analysis


## Setting up python environment

py2.7 (required for psychopy)  
pip install -r requirement.txt

