# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:10:26 2019

@author: yuan
"""
import math 
import numpy as np
def angle2rotm_4x4(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis/np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[ 0.0,     -axis[2],  axis[1]],
                      [ axis[2], 0.0,      -axis[0]],
                      [-axis[1], axis[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M
    
def angle2rotm_3x3(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis/np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array([[ 0.0,     -axis[2],  axis[1]],
                      [ axis[2], 0.0,      -axis[0]],
                      [-axis[1], axis[0],  0.0]])
 
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        R = point - np.dot(R, point)
    return R 
def rotation_calculation_with3x3matrix( vectorBefore , vectorAfter  ) :
    vectorBefore=np.array(vectorBefore)
    vectorAfter=np.array(vectorAfter)
    rotation_axis=np.cross(vectorBefore,vectorAfter)  
    rotationangle=math.acos(np.dot(vectorBefore,vectorAfter)/(np.linalg.norm(vectorBefore)*np.linalg.norm(vectorAfter)))
    matrix=angle2rotm_3x3(rotationangle, rotation_axis)
    return matrix