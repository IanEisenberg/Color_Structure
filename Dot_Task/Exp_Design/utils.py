import cPickle
from glob import glob
# convert from LAB to RGB space
from skimage.color import lab2rgb


def get_difficulties(subject_code):
        motion_file = glob('../Data/RawData/*%s*motion*' % subject_code)[-1]
        color_file = glob('../Data/RawData/*%s*color*' % subject_code)[-1]
        motion_data = cPickle.load(open(motion_file,'r'))
        color_data = cPickle.load(open(color_file,'r'))
        motion_difficulties = {k:v.mean() for k,v in motion_data['trackers'].items()}
        color_difficulties = {k:v.mean() for k,v in color_data['trackers'].items()}
        return motion_difficulties, color_difficulties
    
    
def pixel_lab2rgb(lst):
    lst = [float(x) for x in lst]
    return lab2rgb([[(lst)]]).flatten()
    
