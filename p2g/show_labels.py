# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:34:49 2019

@author: pyni
""" 
import numpy as np    
import pybullet as p  
import random as rd   
import os  
from collision_checker_independent import GraspCollisionChecker
openravechecker=GraspCollisionChecker( view=False)
import dexnet


path_of_stl_file='/media/yuan/pyni/20190410/dexnet/dex-net/data/p_pybullet/'
path_of_tmp_file='/media/yuan/pyni/20190410/dexnet/dex-net/apps/'
path_of_pickle_file='/media/yuan/pyni/20190410/dexnet/'
datasetpath=path_of_tmp_file+str(rd.random())+'temp_database.hdf5'
dexnet_api = dexnet.DexNet() 
try:
   dexnet_api.open_database(datasetpath) 
except:
    pass
try:
    dexnet_api.open_dataset('tmp') 
except:
    pass

dexnet_api.setpath(path_of_pickle_file)
dexnet_api.add_object(path_of_stl_file+'009_gelatin_box.stl',fast=True) 
dexnet_api.showgraspfrompickle('009_gelatin_box', 'ferrari_canny',False,False,openravechecker)
#dexnet_api.showgraspfrompicklewithcandidate ( '009_gelatin_box', 'ferrari_canny', False ,   True ,openravechecker)
 
dexnet_api.close_database( )

os.remove(datasetpath)
 