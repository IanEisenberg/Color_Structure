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

conda env create -f environment.yml
source activate psychopy
pip install -e [Prob_Context_Task directory]

If the enviornment.yml file doesn't work, create a py3.6 environment.
Then you need to install (using conda, preferably):
numpy
pandas
seaborn
scipy
sklearn

Finally,
pip install git+https://github.com/psychopy/psychopy