"""
some util functions
"""
import numpy as np
import pandas as pd

# ********************************************************
# 1st level analysis utility functions
# ********************************************************


def get_beta_series(events_df, regress_rt=True):
    output_dict = {
        'conditions': [],
        'onsets': [],
        'durations': [],
        'amplitudes': []
        }
    for i, row in events_df.iterrows():
        if row.junk == False:
            output_dict['conditions'].append('trial_%s' % str(i+1))
            output_dict['onsets'].append([row.onset])
            output_dict['durations'].append([row.duration])
            output_dict['amplitudes'].append([1])
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')   
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time',
                    subset='junk==False')
    return output_dict
    
def process_confounds(confounds_file):
    """
    scrubbing for TASK
    remove TRs where FD>.5, stdDVARS (that relates to DVARS>.5)
    regressors to use
    ['X','Y','Z','RotX','RotY','RotY','<-firsttemporalderivative','stdDVARs','FD','respiratory','physio','aCompCor0-5']
    junk regressor: errors, ommissions, maybe very fast RTs (less than 50 ms)
    """
    confounds_df = pd.read_csv(confounds_file, sep = '\t', 
                               na_values=['n/a']).fillna(0)
    excessive_movement = (confounds_df.FramewiseDisplacement>.5) & \
                            (confounds_df.stdDVARS>1.2)
    excessive_movement_TRs = excessive_movement[excessive_movement].index
    excessive_movement_regressors = np.zeros([confounds_df.shape[0], 
                                   np.sum(excessive_movement)])
    for i,TR in enumerate(excessive_movement_TRs):
        excessive_movement_regressors[TR,i] = 1
    excessive_movement_regressor_names = ['rejectTR_%d' % TR for TR in 
                                          excessive_movement_TRs]
    # get movement regressors
    movement_regressor_names = ['X','Y','Z','RotX','RotY','RotZ']
    movement_regressors = confounds_df.loc[:,movement_regressor_names]
    movement_regressor_names += ['Xtd','Ytd','Ztd','RotXtd','RotYtd','RotZtd']
    movement_deriv_regressors = np.gradient(movement_regressors,axis=0)
    # add additional relevant regressors
    add_regressor_names = ['FramewiseDisplacement', 'stdDVARS'] 
    add_regressor_names += [i for i in confounds_df.columns if 'aCompCor' in i]
    additional_regressors = confounds_df.loc[:,add_regressor_names].values
    regressors = np.hstack((movement_regressors,
                            movement_deriv_regressors,
                            additional_regressors,
                            excessive_movement_regressors))
    # concatenate regressor names
    regressor_names = movement_regressor_names + add_regressor_names + \
                      excessive_movement_regressor_names
    return regressors, regressor_names
        
def process_physio(cardiac_file, resp_file):
    cardiac_file = '/mnt/temp/sub-s130/ses-1/func/sub-s130_ses-1_task-stroop_run-1_recording-cardiac_physio.tsv.gz'
    resp_file = '/mnt/temp/sub-s130/ses-1/func/sub-s130_ses-1_task-stroop_run-1_recording-respiratory_physio.tsv.gz'
    
    
