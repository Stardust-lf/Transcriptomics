from Utils import get_inter_pos_frames,get_paths,wrapping_simu
import copy
import numpy as np
slice1,slice2 = get_inter_pos_frames(get_paths('Data')[0],get_paths('Data')[1])
slice1_pos = slice1['Position']
slice2_pos = slice2['Position']
scale_max=[min(np.max(slice1['Position'][i][:,0]),np.max(slice2['Position'][i][:,0])) for i in range(len(slice1['Position']))]
scale_min=[max(np.min(slice1['Position'][i][:,0]),np.min(slice2['Position'][i][:,0])) for i in range(len(slice1['Position']))]

scale = list(zip(scale_min,scale_max))
slice1_pos = copy.deepcopy(slice1['Position'])
slice2_pos = copy.deepcopy(slice2['Position'])
values = [slice1_pos[0][i] for i in range(len(slice1_pos[0])) if scale[0][0]<= slice1_pos[0][i][0] <= scale[0][1]]
filled_pos1 = []
filled_pos2 = []
print(len(slice1_pos))
print(len(slice2_pos))
for n in range(len(slice1_pos)):
    leng1 = len(slice1_pos[n])
    leng2 = len(slice2_pos[n])
    if leng1>leng2:
        slice2_pos[n] = np.concatenate([slice2_pos[n],np.zeros(shape=[leng1-leng2,2])],axis=0)
    elif leng1<leng2:
        slice1_pos[n] = np.concatenate([slice1_pos[n],np.zeros(shape=[leng2-leng1,2])],axis=0)
