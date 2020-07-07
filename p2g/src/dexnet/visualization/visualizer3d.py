# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Dex-Net 3D visualizer extension
Author: Jeff Mahler and Jacky Liang
"""
import copy
import json
import IPython
import logging
import numpy as np
import os
try:
    import mayavi.mlab as mv
    import mayavi.mlab as mlab
except:
    logging.info('Failed to import mayavi')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import scipy.spatial.distance as ssd

from autolab_core import RigidTransform, Point
from dexnet.grasping import RobotGripper
from visualization import Visualizer3D

from meshpy import StablePose
import meshpy.obj_file as objf
import meshpy.mesh as m

class DexNetVisualizer3D(Visualizer3D):
    """
    Dex-Net extension of the base Mayavi-based 3D visualization tools
    """
    @staticmethod
    def gripper(gripper, grasp, T_obj_world, color=(0.5, 0.5, 0.5) ):
        """ Plots a robotic gripper in a pose specified by a particular grasp object.

        Parameters
        ----------
        gripper : :obj:`dexnet.grasping.RobotGripper`
            the gripper to plot
        grasp : :obj:`dexnet.grasping.Grasp`
            the grasp to plot the gripper performing
        T_obj_world : :obj:`autolab_core.RigidTransform`
            the pose of the object that the grasp is referencing in world frame
        color : 3-tuple
            color of the gripper mesh
        """
        T_gripper_obj = grasp.gripper_pose(gripper) 
        T_gripper_obj.translation=T_gripper_obj.translation#+sdfcenter[0] +sdforigin[0]
        T_gripper_world = T_obj_world * T_gripper_obj
        T_mesh_world = T_gripper_world * gripper.T_mesh_gripper.inverse()        
        T_mesh_world = T_mesh_world.as_frames('obj', 'world')
      #  print 'T_mesh_world',T_mesh_world
        Visualizer3D.mesh(gripper.mesh.trimesh, T_mesh_world, style='surface', color=color)

    @staticmethod
    def gripper_with_T_gripper_obj(gripper, grasp, T_gripper_obj, color=(0.5, 0.5, 0.5) ):
        """ Plots a robotic gripper in a pose specified by a particular grasp object.

        Parameters
        ----------
        gripper : :obj:`dexnet.grasping.RobotGripper`
            the gripper to plot
        grasp : :obj:`dexnet.grasping.Grasp`
            the grasp to plot the gripper performing
        T_obj_world : :obj:`autolab_core.RigidTransform`
            the pose of the object that the grasp is referencing in world frame
        color : 3-tuple
            color of the gripper mesh
        """
        T_obj_world = RigidTransform(from_frame='obj',to_frame='world')
        T_gripper_world = T_obj_world * T_gripper_obj
        T_mesh_world = T_gripper_world * gripper.T_mesh_gripper.inverse()        
        T_mesh_world = T_mesh_world.as_frames('obj', 'world')
     #   print 'T_mesh_world',T_mesh_world
        Visualizer3D.mesh(gripper.mesh.trimesh, T_mesh_world, style='surface', color=color)

    @staticmethod
    def grasp(grasp, T_obj_world=RigidTransform(from_frame='obj', to_frame='world'),
              tube_radius=0.002, endpoint_color=(0,1,0),
              endpoint_scale=0.004, grasp_axis_color=(0,1,0),sdfcenter=np.array([50.,50.,50.]),sdforigin=np.array([0,0,0])):
        """ Plots a grasp as an axis and center.

        Parameters
        ----------
        grasp : :obj:`dexnet.grasping.Grasp`
            the grasp to plot
        T_obj_world : :obj:`autolab_core.RigidTransform`
            the pose of the object that the grasp is referencing in world frame
        tube_radius : float
            radius of the plotted grasp axis
        endpoint_color : 3-tuple
            color of the endpoints of the grasp axis
        endpoint_scale : 3-tuple
            scale of the plotted endpoints
        grasp_axis_color : 3-tuple
            color of the grasp axis
        """
        g1, g2 = grasp.endpoints
        center = grasp.center#这个center未必是真正的center，所以后面不能用它
       # approach_point = np.array(grasp.approach_point ) 
      #  approach_axis = grasp.rotated_full_axis[:,0]
        #approach_pointfar=approach_point+approach_axis*0.1
      #  print 'approach_angle:',grasp.approach_angle
      #  print 'approach_point:',grasp.approach_point        
        
        g1 = Point(g1, 'obj')
        g2 = Point(g2, 'obj')
        center = Point(center, 'obj')
      #  approach_pointfar2 = Point(approach_pointfar, 'obj')
       # approach_point2 = Point(approach_point, 'obj')        
        
        
        g1_tf = T_obj_world.apply(g1)
        g2_tf = T_obj_world.apply(g2)
        #center_tf = T_obj_world.apply(center)
       # approach_pointfar_tf = T_obj_world.apply(approach_pointfar2)        
       # approach_point2_tf = T_obj_world.apply(approach_point2)  
       # Visualizer3D.points(approach_point2_tf.data, color=(1,0,0), scale=0.001)
        
        
        
        grasp_axis_tf = np.array([g1_tf.data, g2_tf.data])
      #  approach_axis_tf = np.array([approach_point2_tf.data, approach_pointfar_tf.data])        
        #approach_point2_tf_again = np.array([approach_point2_tf.data, approach_point2_tf.data])               
        
        
      #  print [(x[0], x[1], x[2]) for x in grasp_axis_tf] 
        #print [(x[0], x[1], x[2]) for x in grasp_axis_tf] 
        #points = [(x[0]+sdfcenter[0]+sdforigin[0] , x[1]+sdfcenter[1]+sdforigin[1] , x[2] +sdfcenter[2]+sdforigin[2]) for x in grasp_axis_tf]
        points = [(x[0]  , x[1]  , x[2]  ) for x in grasp_axis_tf]
       # approaching = [(y[0]  , y[1]  , y[2]  ) for y in approach_axis_tf]
       # approachingpoint=[(z[0]  , z[1]  , z[2]  ) for z in approach_point2_tf_again]
        
        
        
	#points = [(x[0]  , x[1]  , x[2] ) for x in grasp_axis_tf]
	Visualizer3D.plot3d(points, color=grasp_axis_color, tube_radius=tube_radius)
	#Visualizer3D.plot3d(approaching, color=(0,0.5,0.5), tube_radius=tube_radius) 
    #Visualizer3D.points(approach_point2_tf.data, color=(1,0,0), scale=0.01)
 






    @staticmethod
    def showpoint(surface_point  , T_obj_world=RigidTransform(from_frame='obj', to_frame='world'),color=(0.5,0.5,0),scale=0.001):
        """ Plots a grasp as an axis and center.

        Parameters
        ----------
        grasp : :obj:`dexnet.grasping.Grasp`
            the grasp to plot
        T_obj_world : :obj:`autolab_core.RigidTransform`
            the pose of the object that the grasp is referencing in world frame
        tube_radius : float
            radius of the plotted grasp axis
        endpoint_color : 3-tuple
            color of the endpoints of the grasp axis
        endpoint_scale : 3-tuple
            scale of the plotted endpoints
        grasp_axis_color : 3-tuple
            color of the grasp axis
        """
    
 
        surface_point = Point(surface_point, 'obj')  
        surface_point_tf = T_obj_world.apply(surface_point)
        #center_tf = T_obj_world.apply(center) 
         
	Visualizer3D.points(surface_point_tf.data, color=color, scale=scale)
    #Visualizer3D.points(approach_point2_tf.data, color=(1,0,0), scale=0.01)






    @staticmethod
    def shownormals(normal,surface_point  ,color=(1,0,0),tube_radius=0.001,T_obj_world=RigidTransform(from_frame='obj', to_frame='world')):
        """ Plots a grasp as an axis and center.

        Parameters
        ----------
        grasp : :obj:`dexnet.grasping.Grasp`
            the grasp to plot
        T_obj_world : :obj:`autolab_core.RigidTransform`
            the pose of the object that the grasp is referencing in world frame
        tube_radius : float
            radius of the plotted grasp axis
        endpoint_color : 3-tuple
            color of the endpoints of the grasp axis
        endpoint_scale : 3-tuple
            scale of the plotted endpoints
        grasp_axis_color : 3-tuple
            color of the grasp axis
        """
    
        normalpoint=surface_point+0.02*normal
        
        normalpoint = Point(normalpoint, 'obj')
        surface_point = Point(surface_point, 'obj') 
        
        normal_tf = T_obj_world.apply(normalpoint)
        surface_point_tf = T_obj_world.apply(surface_point) 
        
        normal_axis_tf = np.array([normal_tf.data, surface_point_tf.data])             
        
        
       # print [(x[0], x[1], x[2]) for x in normal_axis_tf] 
        points = [(x[0]  , x[1]  , x[2]  ) for x in normal_axis_tf]  
	Visualizer3D.plot3d(points, color=color , tube_radius=tube_radius) 
    #Visualizer3D.points(approach_point2_tf.data, color=(1,0,0), scale=0.01)
 




    @staticmethod
    def graspwithapproachvectorusingcenter_point(grasp, T_obj_world=RigidTransform(from_frame='obj', to_frame='world'),
              tube_radius=0.001, approaching_color=(0,1,0),
              endpoint_scale=0.004, grasp_axis_color=(0,1,0),sdfcenter=np.array([50.,50.,50.]),sdforigin=np.array([0,0,0])):
        """ Plots a grasp as an axis and center.

        Parameters
        ----------
        grasp : :obj:`dexnet.grasping.Grasp`
            the grasp to plot
        T_obj_world : :obj:`autolab_core.RigidTransform`
            the pose of the object that the grasp is referencing in world frame
        tube_radius : float
            radius of the plotted grasp axis
        endpoint_color : 3-tuple
            color of the endpoints of the grasp axis
        endpoint_scale : 3-tuple
            scale of the plotted endpoints
        grasp_axis_color : 3-tuple
            color of the grasp axis
        """
        g1_, g2_ = grasp.endpoints

        center = grasp.center#这个center未必是真正的center，所以后面不能用它
        approach_point = np.array(grasp.approach_point ) 
        approach_axis = grasp.rotated_full_axis[:,0]
        
        
        g1 = g1_+approach_axis*0.04
        g2 = g2_+approach_axis*0.04        
        approach_pointfar=center+approach_axis*0.1
        approach_pointclose=center+approach_axis*0.04
       # print 'approach_angle:',grasp.approach_angle
      #  print 'approach_point:',grasp.approach_point        
        g1_ = Point(g1_, 'obj')        
        g1 = Point(g1, 'obj')
        g2 = Point(g2, 'obj')
        g2_ = Point(g2_, 'obj')
        center = Point(center, 'obj')
        approach_pointfar2 = Point(approach_pointfar, 'obj')
        approach_pointclose = Point(approach_pointclose, 'obj')
        approach_point2 = Point(approach_point, 'obj')        
        
        
        g1_tf = T_obj_world.apply(g1)
        g2_tf = T_obj_world.apply(g2)
        g1_tf_ = T_obj_world.apply(g1_)
        g2_tf_ = T_obj_world.apply(g2_)        
        
        
        center_tf = T_obj_world.apply(center)
        approach_pointfar_tf = T_obj_world.apply(approach_pointfar2)        
        approach_pointclose_tf = T_obj_world.apply(approach_pointclose)          
        approach_point2_tf = T_obj_world.apply(approach_point2)  
        Visualizer3D.points(approach_point2_tf.data, color=(0,0,0), scale=0.001)
        
        
        
        grasp_axis_tf = np.array([g1_tf.data, g2_tf.data])
        grasp_left_tf = np.array([g1_tf_.data, g1_tf.data])
        grasp_right_tf = np.array([g2_tf_.data, g2_tf.data])
        
        
        approach_axis_tf = np.array([approach_pointclose_tf.data, approach_pointfar_tf.data])        
        #approach_point2_tf_again = np.array([approach_point2_tf.data, approach_point2_tf.data])               
        
        
       # print [(x[0], x[1], x[2]) for x in grasp_axis_tf] 
        #print [(x[0], x[1], x[2]) for x in grasp_axis_tf] 
        #points = [(x[0]+sdfcenter[0]+sdforigin[0] , x[1]+sdfcenter[1]+sdforigin[1] , x[2] +sdfcenter[2]+sdforigin[2]) for x in grasp_axis_tf]
        points = [(x[0]  , x[1]  , x[2]  ) for x in grasp_axis_tf]
        approaching = [(y[0]  , y[1]  , y[2]  ) for y in approach_axis_tf]
        left = [(l[0]  , l[1]  , l[2]  ) for l in grasp_left_tf]
        right = [(r[0]  , r[1]  , r[2]  ) for r in grasp_right_tf]
       # approachingpoint=[(z[0]  , z[1]  , z[2]  ) for z in approach_point2_tf_again]
        
        
        
	#points = [(x[0]  , x[1]  , x[2] ) for x in grasp_axis_tf]
	Visualizer3D.plot3d(points, color=grasp_axis_color, tube_radius=tube_radius)
	Visualizer3D.plot3d(approaching, color=approaching_color, tube_radius=tube_radius) 
 	Visualizer3D.plot3d(left, color=grasp_axis_color, tube_radius=tube_radius) 
  	Visualizer3D.plot3d(right, color=grasp_axis_color, tube_radius=tube_radius) 
    #Visualizer3D.points(approach_point2_tf.data, color=(1,0,0), scale=0.01)
 






    @staticmethod
    def graspwithapproachvectorusingapproach_point(grasp, T_obj_world=RigidTransform(from_frame='obj', to_frame='world'),
              tube_radius=0.001, endpoint_color=(0,1,0),
              endpoint_scale=0.004, grasp_axis_color=(0,1,0),sdfcenter=np.array([50.,50.,50.]),sdforigin=np.array([0,0,0])):
        """ Plots a grasp as an axis and center.

        Parameters
        ----------
        grasp : :obj:`dexnet.grasping.Grasp`
            the grasp to plot
        T_obj_world : :obj:`autolab_core.RigidTransform`
            the pose of the object that the grasp is referencing in world frame
        tube_radius : float
            radius of the plotted grasp axis
        endpoint_color : 3-tuple
            color of the endpoints of the grasp axis
        endpoint_scale : 3-tuple
            scale of the plotted endpoints
        grasp_axis_color : 3-tuple
            color of the grasp axis
        """
        g1, g2 = grasp.endpoints
        center = grasp.center#这个center未必是真正的center，所以后面不能用它
        approach_point = np.array(grasp.approach_point ) 
        approach_axis = grasp.rotated_full_axis[:,0]
        approach_pointfar=approach_point+approach_axis*0.1
       # print 'approach_angle:',grasp.approach_angle
      #  print 'approach_point:',grasp.approach_point        
        
        g1 = Point(g1, 'obj')
        g2 = Point(g2, 'obj')
        center = Point(center, 'obj')
        approach_pointfar2 = Point(approach_pointfar, 'obj')
        approach_point2 = Point(approach_point, 'obj')        
        
        
        g1_tf = T_obj_world.apply(g1)
        g2_tf = T_obj_world.apply(g2)
        #center_tf = T_obj_world.apply(center)
        approach_pointfar_tf = T_obj_world.apply(approach_pointfar2)        
        approach_point2_tf = T_obj_world.apply(approach_point2)  
        Visualizer3D.points(approach_point2_tf.data, color=(1,0,0), scale=0.001)
        
        
        
        grasp_axis_tf = np.array([g1_tf.data, g2_tf.data])
        approach_axis_tf = np.array([approach_point2_tf.data, approach_pointfar_tf.data])        
        #approach_point2_tf_again = np.array([approach_point2_tf.data, approach_point2_tf.data])               
        
        
       # print [(x[0], x[1], x[2]) for x in grasp_axis_tf] 
        #print [(x[0], x[1], x[2]) for x in grasp_axis_tf] 
        #points = [(x[0]+sdfcenter[0]+sdforigin[0] , x[1]+sdfcenter[1]+sdforigin[1] , x[2] +sdfcenter[2]+sdforigin[2]) for x in grasp_axis_tf]
        points = [(x[0]  , x[1]  , x[2]  ) for x in grasp_axis_tf]
        approaching = [(y[0]  , y[1]  , y[2]  ) for y in approach_axis_tf]
       # approachingpoint=[(z[0]  , z[1]  , z[2]  ) for z in approach_point2_tf_again]
        
        
        
	#points = [(x[0]  , x[1]  , x[2] ) for x in grasp_axis_tf]
	Visualizer3D.plot3d(points, color=grasp_axis_color, tube_radius=tube_radius)
	Visualizer3D.plot3d(approaching, color=(0,0.5,0.5), tube_radius=tube_radius) 
    #Visualizer3D.points(approach_point2_tf.data, color=(1,0,0), scale=0.01)
 
 
    @staticmethod
    def gripper_on_object(gripper, grasp, obj, stable_pose=None,
                          T_table_world=RigidTransform(from_frame='table', to_frame='world'),
                          gripper_color=(0.5,0.5,0.5), object_color=(0.5,0.5,0.5),
                          style='surface', plot_table=True, table_dim=0.15 ):
        """ Visualize a gripper on an object.
        
        Parameters
        ----------
        gripper : :obj:`dexnet.grasping.RobotGripper`
            gripper to plot
        grasp : :obj:`dexnet.grasping.Grasp`
            grasp to plot the gripper in
        obj : :obj:`dexnet.grasping.GraspableObject3D`
            3D object to plot the gripper on
        stable_pose : :obj:`autolab_core.RigidTransform`
            stable pose of the object on a planar worksurface
        T_table_world : :obj:`autolab_core.RigidTransform`
            pose of table, specified as a transformation from mesh frame to world frame
        gripper_color : 3-tuple
            color of the gripper mesh
        object_color : 3-tuple
            color of the object mesh
        style : :obj:`str`
            color of the object mesh
        plot_table : bool
            whether or not to plot the table
        table_dim : float
            dimension of the table
        """
#        if stable_pose is None:
#            Visualizer3D.mesh(obj.mesh.trimesh, color=object_color, style=style)
        T_obj_world = RigidTransform(from_frame='obj',
                                         to_frame='world')
#        else:
#            T_obj_world = Visualizer3D.mesh_stable_pose(obj.mesh.trimesh, stable_pose, T_table_world=T_table_world, color=object_color, style=style, plot_table=plot_table, dim=table_dim)
#            
            
      #  print      'T_obj_world',T_obj_world       
        DexNetVisualizer3D.gripper(gripper, grasp, T_obj_world, color=gripper_color )

