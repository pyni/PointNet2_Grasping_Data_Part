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
Revised from dexnet  (https://berkeleyautomation.github.io/dex-net/)
@author: pyni
""" 
import collections
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tempfile
import pickle
import random as rd
#import open3d
# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from autolab_core import YamlConfig, RigidTransform
import random as rd
#TODO
#Once trimesh integration is here via meshpy remove this
import trimesh
from dexnet.grasping import  RobotGripper,Contact3D, ParallelJawPtGrasp3D
import dexnet.database.database as db
import dexnet.grasping.grasp_quality_config as gqc
import dexnet.grasping.grasp_quality_function as gqf
import dexnet.grasping.grasp_sampler as gs
import dexnet.grasping.gripper as gr
import dexnet.database.mesh_processor as mp 
from rotation_calculation import  rotation_calculation_with3x3matrix
from meshpy import convex_decomposition, Mesh3D
try:
    from dexnet.visualization import DexNetVisualizer3D as vis
except:
    logger.warning('Failed to import DexNetVisualizer3D, visualization methods will be unavailable')
from dexnet.grasping import Contact3D
DEXNET_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../../') + '/'
DEXNET_API_DEFAULTS_FILE = DEXNET_DIR + 'cfg/api_defaults.yaml' 
class DexNet(object):
    """Class providing an interface for main DexNet pipeline
    
    Attributes
    ----------
    database : :obj:`dexnet.database.Database
        Current active database. Can set manually, or with open_database
    dataset : :obj:`dexnet.database.Dataset
        Current active dataset. Can set manually, or with open_dataset
    default_config : :obj:`dictionary`
        A dictionary containing default config values
        See Other Parameters for details. These parameters are also listed under the function(s) they are relevant to
        Also, see:
            dexnet.grasping.grasp_quality_config for metrics and their associated configs
            dexnet.database.mesh_processor for configs associated with initial mesh processing
        
    Other Parameters
    ----------------
    cache_dir 
        Cache directory for to store intermediate files. If None uses a temporary directory
    use_default_mass
        If True, clobbers mass and uses default_mass as mass always
    default_mass
        Default mass value if mass is not given, or if use_default_mass is set
    gripper_dir
        Directory where the grippers models and parameters are
    metric_display_rate
        Number of grasps to compute metrics for before logging a line
    gravity_accel
        Gravity acceleration for computing gravity-based metrics
    metrics
        Dictionary mapping metric names to metric config dicts
        For available metrics and their config parameters see dexnet.grasping.grasp_quality_config
    grasp_sampler
        type of grasp sampler to use. One of {'antipodal', 'gaussian', 'uniform'}.
    max_grasp_sampling_iters
        number of attempts to return an exact number of grasps before giving up
    export_format
        Format for export. One of obj, stl, urdf
    export_scale
        Scale for export.
    export_overwrite
        If True, will overwrite existing files
    animate
        Whether or not to animate the displayed object
    quality_scale
        Range to scale quality metric values to
    show_gripper
        Whether or not to show the gripper in the visualization
    min_metric
        lowest value of metric to show grasps for
    max_plot_gripper
        Number of grasps to plot
    """
    def __init__(self):
        """Create a DexNet object
        """
        self.database = None
        self.dataset = None
        
        self._database_temp_cache_dir = None
        self.path_of_pickle_file=None
        # open default config
        self.default_config = YamlConfig(DEXNET_API_DEFAULTS_FILE)
        # Resolve gripper_dir and cache_dir relative to dex-net root
        self.default_config['cache_dir'] = None
        for key in ['gripper_dir']:
            if not os.path.isabs(self.default_config[key]):
                self.default_config[key] = os.path.realpath(DEXNET_DIR + self.default_config[key])
    
    #TODO
    #Move to YamlConfig
    @staticmethod
    def _deep_update_config(config, updates):
        """ Deep updates a config dict """
        for key, value in updates.iteritems():
            if isinstance(value, collections.Mapping):
                config[key] = DexNet._deep_update_config(config.get(key, {}), value)
            else:
                config[key] = value
        return config
    
    def _get_config(self, updates=None):
        """ Gets a copy of the default config dict with updates from the dict passed in applied """
        updated_cfg = copy.deepcopy(self.default_config.config)
        if updates is not None:
            DexNet._deep_update_config(updated_cfg, updates)
        return updated_cfg
    
    def _check_opens(self):
        """ Checks that database and dataset are open """
        if self.database is None:
            raise RuntimeError('You must open a database first')
        if self.dataset is None:
            raise RuntimeError('You must open a dataset first')
        
    def open_database(self, database_path, config=None, create_db=True):
        """Open/create a database.

        Parameters
        ----------
        database_path : :obj:`str`
            Path (can be relative) to the database, or the path to create a database at.        
        create_db : boolean
            If True, creates database if one does not exist at location specified.
            If False, raises error if database does not exist at location specified.
        config : :obj:`dict`
            Dictionary of parameters for database creation
            Parameters are in Other Parameters. Values from self.default_config are used for keys not provided.
            
        Other Parameters
        ----------------
        cache_dir 
            Cache directory for to store intermediate files. If None uses a temporary directory
            
        Raises
        ------
        ValueError
            If database_path does not have an extension corresponding to a hdf5 database.
            If database does not exist at path and create_db is False.
        """
        config = self._get_config(config)
        
        if self.database is not None:
            if self._database_temp_cache_dir is not None:
                shutil.rmtree(self._database_temp_cache_dir)
                self._database_temp_cache_dir = None
            self.database.close()

        # Check database path extension
        _, database_ext = os.path.splitext(database_path)
        if database_ext != db.HDF5_EXT:
            raise ValueError('Database must have extension {}'.format(db.HDF5_EXT)) 

        # Abort if database does not exist and create_db is False
        if not os.path.exists(database_path):
            if not create_db:
                raise ValueError('Database does not exist at path {} and create_db is False'.format(database_path))
            else:
                logger.info("File not found, creating new database at {}".format(database_path))
                
        # Create temp dir if cache dir is not provided
        cache_dir = config['cache_dir']
        if cache_dir is None:
            cache_dir = tempfile.mkdtemp()
            self._database_temp_cache_dir = cache_dir
            
        # Open database
        self.database = db.Hdf5Database(database_path,access_level=db.READ_WRITE_ACCESS,cache_dir=cache_dir)
    #cc=self.database.dataset('mini_dexnet')
        #dd=cc['vase']            sdf.surface_points(grid_basis=False)
    def open_dataset(self, dataset_name, config=None, create_ds=True):
        """Open/create a dataset

        Parameters
        ----------
        dataset_name : :obj:`str`
            Name of dataset to open/create
        create_ds : boolean
            If True, creates dataset if one does not exist with name specified.
            If False, raises error if specified dataset does not exist
        config : :obj:`dict`
            Dictionary containing a key 'metrics' that maps to a dictionary mapping metric names to metric config dicts
            For available metrics and their corresponding config parameters see dexnet.grasping.grasp_quality_config
            Values from self.default_config are used for keys not provided
        
        Raises
        ------
        ValueError
            If dataset_name is invalid. Also if dataset does not exist and create_ds is False
        RuntimeError
            No database open
        """
        if self.database is None:
            raise RuntimeError('You must open a database first')
        
        config = self._get_config(config)
        
        tokens = dataset_name.split()
        if len(tokens) > 1:
            raise ValueError("dataset_name \"{}\" is invalid (contains delimiter)".format(dataset_name))
            
        existing_datasets = [d.name for d in self.database.datasets]
            
        # create/open new ds
        if dataset_name not in existing_datasets:
            if create_ds:
                logger.info("Creating new dataset {}".format(dataset_name))
                self.database.create_dataset(dataset_name)
                self.dataset = self.database.dataset(dataset_name)
                metric_dict = config['metrics']
                for metric_name, metric_spec in metric_dict.iteritems():
                    # create metric
                    metric_config = gqc.GraspQualityConfigFactory.create_config(metric_spec)            
                    self.dataset.create_metric(metric_name, metric_config)
            else:
                raise ValueError(
                    "dataset_name \"{}\" is invalid (does not exist, and create_ds is False)".format(dataset_name))
        else:
            self.dataset = self.database.dataset(dataset_name)
            
        if self.dataset.metadata is None:
            self._attach_metadata()
    
    #TODO
    #Once trimesh integration is here via meshpy remove this
    @staticmethod
    def _meshpy_to_trimesh(mesh_m3d):
        vertices = mesh_m3d.vertices
        faces = mesh_m3d.triangles
        mesh_tm = trimesh.Trimesh(vertices, faces)
        return mesh_tm
    #TODO
    #Once trimesh integration is here via meshpy remove this
    @staticmethod
    def _trimesh_to_meshpy(mesh_tm):
        vertices = mesh_tm.vertices
        triangles = mesh_tm.faces
        mesh_m3d = Mesh3D(vertices, triangles)
        return mesh_m3d
    #TODO
    #Once trimesh integration is here via meshpy remove this
    @staticmethod
    def is_watertight(mesh):
        mesh_tm = DexNet._meshpy_to_trimesh(mesh)
        return mesh_tm.is_watertight
    
    #TODO
    #Make this better and more general
    def _attach_metadata(self):
        """ Attach default metadata to dataset. Currently only watertightness and number of connected components, and
        only watertightness has an attached function.
        """
        self.dataset.create_metadata("watertightness", "float", "1.0 if the mesh is watertight, 0.0 if it is not")
        self.dataset.attach_metadata_func("watertightness", DexNet.is_watertight, overwrite=False, store_func=True)
        self.dataset.create_metadata("num_con_comps", "float", "Number of connected components (may not be watertight) in the mesh")
        self.dataset.attach_metadata_func("num_con_comps", object(), overwrite=False, store_func=True)
    
    def add_object(self, filepath, config=None, mass=None, name=None,fast=False):
        """Add graspable object to current open dataset
        
        Parameters
        ----------
        filepath : :obj:`str`
            Path to mesh file
        config : :obj:`dict`
            Dictionary of parameters for mesh creating/processing
            Parameters are in Other parameters. Values from self.default_config are used for keys not provided.
            See dexnet.database.mesh_processor.py for details on the parameters available for mesh processor
        name : :obj:`str`
            Name to use for graspable. If None defaults to the name of the obj file provided in filepath
        mass : float
            Mass of object. Gets clobbered if use_default_mass is set in config.
            
        Other Parameters
        ----------------
        cache_dir 
            Cache directory for mesh processor to store intermediate files. If None uses a temporary directory
        use_default_mass
            If True, clobbers mass and uses default_mass as mass always
        default_mass
            Default mass value if mass is not given, or if use_default_mass is set
        
        Raises
        ------
        RuntimeError
            Graspable with same name already in database.
            Database or dataset not opened.
        """
        self._check_opens()
        config = self._get_config(config)
        
        if name is None:
            _, root = os.path.split(filepath)
            name, _ = os.path.splitext(root)
        if name in self.dataset.object_keys:
            raise RuntimeError('An object with key %s already exists. ' +
                               'Delete the object with delete_graspable first if replacing it'.format(name))
        
        if mass is None or config['use_default_mass']:
            mass = config['default_mass']
         
        # Create temp dir if cache dir is not provided
        mp_cache = config['cache_dir']
        del_cache = False
        if mp_cache is None:
            mp_cache = tempfile.mkdtemp()
            del_cache = True
        
        # open mesh preprocessor
        mesh_processor = mp.MeshProcessor(filepath, mp_cache)
 
        mesh_processor.generate_graspable(config,fast)
    
 
       
        # write to database
        self.dataset.create_graspable(name, mesh_processor.mesh, mesh_processor.sdf,
                                      mesh_processor.stable_poses,
                                      mass=mass)

        # Delete cache if using temp cache
        if del_cache:
            shutil.rmtree(mp_cache)
 
    def setpath(self,path_of_pickle_file):
        self.path_of_pickle_file=path_of_pickle_file 
          
 





    def readdatafrompicklewithcandidate(self, object_name):         
        f_labelgrasps=open(self.path_of_pickle_file+'labelgrasps_'+object_name+'_'+'baxter.pk','r')
        f_labelmetrics=open(self.path_of_pickle_file+'labelmetrics_'+object_name+'_'+'baxter.pk','r')
        f_candidate_labelgrasps=open(self.path_of_pickle_file+'candidatelabelgrasps_'+object_name+'_'+'baxter.pk','r')
        f_candidate_labelmetrics=open(self.path_of_pickle_file+'candidatelabelmetrics_'+object_name+'_'+'baxter.pk','r')        
        
        #print 'start'
        labelgrasps=pickle.load(f_labelgrasps)
        labelmetrics=pickle.load(f_labelmetrics)            
        candidate_labelgrasps=pickle.load(f_candidate_labelgrasps)
        candidate_labelmetrics=pickle.load(f_candidate_labelmetrics)            
        
        #print 'end'       
        return labelgrasps,labelmetrics,candidate_labelgrasps,candidate_labelmetrics



       
    def readdatafrompickle(self, object_name):         
        f_labelgrasps=open(self.path_of_pickle_file+'labelgrasps_'+object_name+'_'+'baxter.pk','r')
        f_labelmetrics=open(self.path_of_pickle_file+'labelmetrics_'+object_name+'_'+'baxter.pk','r')
        labelgrasps=pickle.load(f_labelgrasps)
        labelmetrics=pickle.load(f_labelmetrics)               
        return labelgrasps,labelmetrics
        
    def showgraspfrompickle(self, object_name,metric_name,onlynagative,onlypositive,openravechecker):    
        object = self.dataset[object_name] 
        labelgrasps,labelmetrics=self.readdatafrompickle(object_name)

        vis.figure()
        vis.mesh(object.mesh.trimesh ,style='surface')
        config=self._get_config(None)
        low =0.0  
        high = np.max([labelmetrics[i][metric_name] for i in range(len(labelmetrics))])
     #   print 'high',high,config['quality_scale']
        if low == high:
            q_to_c = lambda quality: config['quality_scale']
        else:
            
            q_to_c = lambda quality: config['quality_scale'] * (quality - low) / (high - low)

        recalculate_grasp=[]    
        for i in range(len(labelgrasps)):
                 
                        if labelmetrics[i][metric_name]==0 :
                           if not onlypositive:
                               vis.graspwithapproachvectorusingcenter_point(labelgrasps[i] ,approaching_color=(1,0,0), grasp_axis_color=(1,0,0) )
                        elif  labelmetrics[i][metric_name]==-1   :
                            if not onlypositive:
                                 vis.shownormals(labelgrasps[i][1],labelgrasps[i][0],color=(1,0,0) )
                                 recalculate_grasp.append(labelgrasps[i])
                        elif  labelmetrics[i][metric_name]>0   :
                                if not onlynagative:
                                    color = plt.get_cmap('hsv')(q_to_c(labelmetrics[i][metric_name]))[:-1] 
                                    vis.graspwithapproachvectorusingcenter_point(labelgrasps[i] ,approaching_color=color, grasp_axis_color=color )
                                recalculate_grasp.append(labelgrasps[i])
                        print i   # ,high
                                
        vis.pose(RigidTransform(), alpha=0.1)
        vis.show(animate=False)       
 
 

      
    def showgraspfrompicklewithcandidate(self, object_name,metric_name,onlynagative,onlypositive,openravechecker):    
        object = self.dataset[object_name] 
        labelgrasps,labelmetrics,candidate_labelgrasps,candidate_labelmetrics=self.readdatafrompicklewithcandidate(object_name)


        config=self._get_config(None)
        low =0.0  #np.min(metrics)
        high = np.max([labelmetrics[i][metric_name] for i in range(len(labelmetrics))])
       # print 'high',high,config['quality_scale']
        if low == high:
            q_to_c = lambda quality: config['quality_scale']
        else:
            
            q_to_c = lambda quality: config['quality_scale'] * (quality - low) / (high - low)

        recalculate_grasp=[]    
        vis.figure()
        vis.mesh(object.mesh.trimesh ,style='surface')
        for i in range(len(labelgrasps)): 
         
                        if labelmetrics[i][metric_name]==0 : 
                           if not onlypositive: 
                                    vis.figure()
                                    vis.mesh(object.mesh.trimesh ,style='surface')
                                    color = plt.get_cmap('hsv')(q_to_c(labelmetrics[i][metric_name]))[:-1] 
                                    
                                    vis.graspwithapproachvectorusingcenter_point(labelgrasps[i] ,approaching_color=(1,0,0), grasp_axis_color=(1,0,0) )
                                    for kk in range(len(candidate_labelgrasps[i] )):  
                                        print 'candidate_labelgrasps[i][kk]',candidate_labelgrasps[i][kk]
                                        color = plt.get_cmap('hsv')(q_to_c(candidate_labelmetrics[i][candidate_labelgrasps[i][kk].id][metric_name]))[:-1] 
                                        vis.graspwithapproachvectorusingcenter_point(candidate_labelgrasps[i][kk] ,approaching_color=color, grasp_axis_color=color )
                         
                                    recalculate_grasp.append(labelgrasps[i])
                                    print 'maxtrics',i,labelmetrics[i][metric_name]   # ,high
                                
                                    vis.pose(RigidTransform(), alpha=0.1) 
                    
                                    vis.show(animate=False)             
                        elif  labelmetrics[i][metric_name]==-1   :
                            if not onlypositive:
                                 vis.shownormals(labelgrasps[i][1],labelgrasps[i][0],color=(0,0,1) ) 
                        else:
                                if not onlynagative: 
                                    vis.figure()
                                    vis.mesh(object.mesh.trimesh ,style='surface')
                                    color = plt.get_cmap('hsv')(q_to_c(labelmetrics[i][metric_name]))[:-1]                               
                                    if len(candidate_labelgrasps[i])>0:
                                        A=np.array(labelgrasps[i].rotated_full_axis[:,0] )
                                        B=np.array(candidate_labelgrasps[i][0].rotated_full_axis[:,0] ) 
                                        num = np.dot(A.T, B)
                                        print num
                                        if num<0.99:
                                            labelgrasps[i].approach_angle_= labelgrasps[i]._angle_aligned_with_table( -candidate_labelgrasps[i][0].rotated_full_axis[:,0])
                                                                        
                                    vis.graspwithapproachvectorusingcenter_point(labelgrasps[i] ,approaching_color=color, grasp_axis_color=(0,0,1) )
                                    print 'length',len(candidate_labelgrasps[i] )
                                    for kk in range(len(candidate_labelgrasps[i] )):   
                                        color = plt.get_cmap('hsv')(q_to_c(candidate_labelmetrics[i][candidate_labelgrasps[i][kk].id][metric_name]))[:-1] 
                                        vis.graspwithapproachvectorusingcenter_point(candidate_labelgrasps[i][kk] ,approaching_color=(0,1,1), grasp_axis_color=color )
 
                                    recalculate_grasp.append(labelgrasps[i])
                                    print 'maxtrics',i,labelmetrics[i][metric_name]   # ,high
                                
                                    vis.pose(RigidTransform(), alpha=0.1) 
                                    vis.show(animate=False) 
 
    def delete_object(self, object_name):
        """ Delete an object
        
        Parameters
        ----------
        object_name : :obj:`str`
            Object to delete
        
        Raises
        ------
        ValueError
            invalid object name
        RuntimeError
            Database or dataset not opened
        """
        self._check_opens()
        if object_name not in self.dataset.object_keys:
            raise ValueError("{} is not a valid object name".format(object_name))

        logger.info('Deleting {}'.format(object_name))
        self.dataset.delete_graspable(object_name)
        
    def close_database(self):
        if self.database:
            logger.info('Closing database')
            self.database.close()
        # Delete cache if using temp cache
        if self._database_temp_cache_dir is not None:
            shutil.rmtree(self._database_temp_cache_dir)
            
            