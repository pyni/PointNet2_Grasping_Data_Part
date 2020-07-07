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
Quasi-static point-based grasp quality metrics.
Author: Jeff Mahler and Brian Hou
"""
import logging
import numpy as np
try:
    import pyhull.convex_hull as cvh
except:
    logging.warning('Failed to import pyhull')
try:
    import cvxopt as cvx
except:
    logging.warning('Failed to import cvx')
import os
import scipy.spatial as ss
import sys
import time

from dexnet.grasping import PointGrasp, GraspableObject3D, GraspQualityConfig

import meshpy.obj_file as obj_file
import meshpy.sdf_file as sdf_file

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import mayavi.mlab as mv
except:
    logging.warning('Failed to import mayavi')
import IPython

# turn off output logging
#cvx.solvers.options['show_progress'] = False

class PointGraspMetrics3D:
    """ Class to wrap functions for quasistatic point grasp quality metrics.
    """

    @staticmethod
    def grasp_quality(grasp, obj, params, vis = False):
        """
        Computes the quality of a two-finger point grasps on a given object using a quasi-static model.

        Parameters
        ----------
        grasp : :obj:`ParallelJawPtGrasp3D`
            grasp to evaluate
        obj : :obj:`GraspableObject3D`
            object to evaluate quality on
        params : :obj:`GraspQualityConfig`
            parameters of grasp quality function
        """
        start = time.time()
        if not isinstance(grasp, PointGrasp):
            raise ValueError('Must provide a point grasp object')
        if not isinstance(obj, GraspableObject3D):
            raise ValueError('Must provide a 3D graspable object')
        if not isinstance(params, GraspQualityConfig):
            raise ValueError('Must provide GraspQualityConfig')

        # read in params
        method = params.quality_method
        friction_coef = params.friction_coef
        num_cone_faces = params.num_cone_faces
        soft_fingers = params.soft_fingers
        check_approach = params.check_approach
        if not hasattr(PointGraspMetrics3D, method):
            raise ValueError('Illegal point grasp metric %s specified' %(method))
       # print 'friction_coeffriction_coeffriction_coeffriction_coeffriction_coef',friction_coef
        # get point grasp contacts
        contacts_start = time.time()
        if grasp.contact_==[]:
            contacts_found, contacts = grasp.close_fingers(obj, check_approach=check_approach, vis=vis)
        else: 
            # print 'fast'
             contacts = grasp.contact_
             contacts_found= True
        if not contacts_found:
            print 'Contacts not found'
            logging.debug('Contacts not found')
            return -1

        if method == 'force_closure':
            # Use fast force closure test (Nguyen 1988) if possible.
            if len(contacts) == 2:
                c1, c2 = contacts
                return PointGraspMetrics3D.force_closure(c1, c2, friction_coef)

            # Default to QP force closure test.
            method = 'force_closure_qp'

        # add the forces, torques, etc at each contact point
        forces_start = time.time()
        num_contacts = len(contacts)
        forces = np.zeros([3,0])
        torques = np.zeros([3,0])
        normals = np.zeros([3,0])
        for i in range(num_contacts):
            contact = contacts[i]
            if vis:
                if i == 0:
                    contact.plot_friction_cone(color='y')
                else:
                    contact.plot_friction_cone(color='c')

            # get contact forces
            force_success, contact_forces, contact_outward_normal = contact.friction_cone(num_cone_faces, friction_coef)

            if not force_success:
                #print 'Force computation failed'
                logging.debug('Force computation failed')
                if params.all_contacts_required:
                    return -2
           # print('force_success')  
            # get contact torques
            torque_success, contact_torques = contact.torques(contact_forces)
            if not torque_success:
                print 'Torque computation failed'
                logging.debug('Torque computation failed')
                if params.all_contacts_required:
                    return -3
          #  print('torque_success')  
            # get the magnitude of the normal force that the contacts could apply
            n = contact.normal_force_magnitude()  #n为力在法线上的分量 

            forces = np.c_[forces, n * contact_forces]#这难道是因为contact_forces是假设法线力为1来算的，所以这边要乘以一个正确的法线幅值
            torques = np.c_[torques, n * contact_torques]
            normals = np.c_[normals, n * -contact_outward_normal] # store inward pointing normals   这里把物体表面的法线调了一个方向，（朝内了）


        if normals.shape[1] == 0:
            print 'No normals'
            logging.debug('No normals')
            return -4

        # normalize torques
        if 'torque_scaling' not in params.keys():
            torque_scaling = 1.0
            if method == 'ferrari_canny_L1':
                mn, mx = obj.mesh.bounding_box()
              #  print 'mn, mx',mn, mx
                torque_scaling = 1.0 / np.median(mx)
            params.torque_scaling = torque_scaling 
           # print 'torque_scaling :',torque_scaling 
        if vis:
            ax = plt.gca()
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.show()
       # print('preprocess success')  
        # evaluate the desired quality metric
        quality_start = time.time()
        Q_func = getattr(PointGraspMetrics3D, method)
        quality = Q_func(forces, torques, normals,
                         soft_fingers=soft_fingers,
                         params=params)

        end = time.time()
        logging.debug('Contacts took %.3f sec' %(forces_start - contacts_start))
        logging.debug('Forces took %.3f sec' %(quality_start - forces_start))
        logging.debug('Quality eval took %.3f sec' %(end - quality_start))
        logging.debug('Everything took %.3f sec' %(end - start))

        return quality

    @staticmethod
    def grasp_matrix(forces, torques, normals, soft_fingers=False,
                     finger_radius=0.005, params=None):
        """ Computes the grasp map between contact forces and wrenchs on the object in its reference frame.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        finger_radius : float
            the radius of the fingers to use
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        G : 6xM :obj:`numpy.ndarray`
            grasp map
        """
        if params is not None and 'finger_radius' in params.keys():
            finger_radius = params.finger_radius
        num_forces = forces.shape[1]
        num_torques = torques.shape[1]
        if num_forces != num_torques:
            raise ValueError('Need same number of forces and torques')

        num_cols = num_forces
        if soft_fingers:
            num_normals = 2
            if normals.ndim > 1:
                num_normals = 2*normals.shape[1]
            num_cols = num_cols + num_normals

        G = np.zeros([6, num_cols])
        for i in range(num_forces):
            G[:3,i] = forces[:,i]
          #  print 'forces[:,i]:',forces[:,i]
            G[3:,i] = params.torque_scaling * torques[:,i]
      #  print 'params.torque_scaling',params.torque_scaling
        #    print 'torques[:,i]:', params.torque_scaling * torques[:,i],  params.torque_scaling 
        if soft_fingers:
            torsion = np.pi * finger_radius**2 * params.friction_coef * normals * params.torque_scaling
            pos_normal_i = -num_normals
            neg_normal_i = -num_normals + num_normals / 2
            G[3:,pos_normal_i:neg_normal_i] = torsion
            G[3:,neg_normal_i:] = -torsion#?????????????????why????

        return G

    @staticmethod
    def force_closure(c1, c2, friction_coef, use_abs_value=True):
        """" Checks force closure using the antipodality trick.

        Parameters
        ----------
        c1 : :obj:`Contact3D`
            first contact point
        c2 : :obj:`Contact3D`
            second contact point
        friction_coef : float
            coefficient of friction at the contact point
        use_abs_value : bool
            whether or not to use directoinality of the surface normal (useful when mesh is not oriented)

        Returns
        -------
        int : 1 if in force closure, 0 otherwise
        """
        if c1.point is None or c2.point is None or c1.normal is None or c2.normal is None:
            return 0
        p1, p2 = c1.point, c2.point
        n1, n2 = -c1.normal, -c2.normal # inward facing normals

        if (p1 == p2).all(): # same point
            return 0

        for normal, contact, other_contact in [(n1, p1, p2), (n2, p2, p1)]:
            diff = other_contact - contact
            normal_proj = normal.dot(diff) / np.linalg.norm(normal)
            if use_abs_value:
                normal_proj = abs(normal.dot(diff)) / np.linalg.norm(normal)

            if normal_proj < 0:
                return 0 # wrong side
            alpha = np.arccos(normal_proj / np.linalg.norm(diff))
            if alpha > np.arctan(friction_coef):
                return 0 # outside of friction cone
        return 1

    @staticmethod
    def force_closure_qp(forces, torques, normals, soft_fingers=False,
                         wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                         params=None):
        """ Checks force closure by solving a quadratic program (whether or not zero is in the convex hull)

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        int : 1 if in force closure, 0 otherwise
        """
        if params is not None:
            if 'wrench_norm_thresh' in params.keys():
                wrench_norm_thresh = params.wrench_norm_thresh
            if 'wrench_regularizer' in params.keys():
                wrench_regularizer = params.wrench_regularizer

        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers, params=params)
        min_norm, _ = PointGraspMetrics3D.min_norm_vector_in_facet(G, wrench_regularizer=wrench_regularizer)
        return 1 * (min_norm < wrench_norm_thresh) # if greater than wrench_norm_thresh, 0 is outside of hull

    @staticmethod
    def partial_closure(forces, torques, normals, soft_fingers=False,
                        wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                        params=None):
        """ Evalutes partial closure: whether or not the forces and torques can resist a specific wrench.
        Estimates resistance by sollving a quadratic program (whether or not the target wrench is in the convex hull).

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        int : 1 if in partial closure, 0 otherwise
        """
        force_limit = None
        if params is None:
            return 0
        force_limit = params.force_limits
        target_wrench = params.target_wrench
        if 'wrench_norm_thresh' in params.keys():
            wrench_norm_thresh = params.wrench_norm_thresh
        if 'wrench_regularizer' in params.keys():
            wrench_regularizer = params.wrench_regularizer

        # reorganize the grasp matrix for easier constraint enforcement in optimization
        num_fingers = normals.shape[1]
        num_wrenches_per_finger = forces.shape[1] / num_fingers
        G = np.zeros([6,0])
        for i in range(num_fingers):
            start_i = num_wrenches_per_finger * i
            end_i = num_wrenches_per_finger * (i + 1)
            G_i = PointGraspMetrics3D.grasp_matrix(forces[:,start_i:end_i], torques[:,start_i:end_i], normals[:,i:i+1],
                                                   soft_fingers, params=params)
            G = np.c_[G, G_i]

        wrench_resisted, _ = PointGraspMetrics3D.wrench_in_positive_span(G, target_wrench, force_limit, num_fingers,
                                                                wrench_norm_thresh=wrench_norm_thresh,
                                                                wrench_regularizer=wrench_regularizer)
        return 1 * wrench_resisted

    @staticmethod
    def wrench_resistance(forces, torques, normals, soft_fingers=False, 
                          wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                          finger_force_eps=1e-9, params=None):
        """ Evalutes wrench resistance: the inverse norm of the contact forces required to resist a target wrench
        Estimates resistance by sollving a quadratic program (min normal contact forces to produce a wrench).

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        finger_force_eps : float
            small float to prevent numeric issues in wrench resistance metric
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of wrench resistance metric
        """
        force_limit = None
        if params is None:
            return 0
        force_limit = params.force_limits
        target_wrench = params.target_wrench
       # print 'target_wrench',target_wrench
        if 'wrench_norm_thresh' in params.keys():
            wrench_norm_thresh = params.wrench_norm_thresh
        if 'wrench_regularizer' in params.keys():
            wrench_regularizer = params.wrench_regularizer
        if 'finger_force_eps' in params.keys():
            finger_force_eps = params.finger_force_eps

        # reorganize the grasp matrix for easier constraint enforcement in optimization
        num_fingers = normals.shape[1]
        num_wrenches_per_finger = forces.shape[1] / num_fingers #也就说一个摩擦锥的离散维度
        G = np.zeros([6,0])
        for i in range(num_fingers):
            start_i = num_wrenches_per_finger * i
            end_i = num_wrenches_per_finger * (i + 1)
            G_i = PointGraspMetrics3D.grasp_matrix(forces[:,start_i:end_i], torques[:,start_i:end_i], normals[:,i:i+1],soft_fingers, params=params)
            G = np.c_[G, G_i] 
        # compute metric from finger force norm
        Q = 0
        wrench_resisted, finger_force_norm = PointGraspMetrics3D.wrench_in_positive_span(G, target_wrench, force_limit, num_fingers,
                                                                                         wrench_norm_thresh=wrench_norm_thresh,
                                                                                         wrench_regularizer=wrench_regularizer)
        if wrench_resisted:
            Q = 1.0 / (finger_force_norm + finger_force_eps) - 1.0 / (2 * force_limit)
        return Q

    @staticmethod
    def min_singular(forces, torques, normals, soft_fingers=False, params=None):
        """ Min singular value of grasp matrix - measure of wrench that grasp is "weakest" at resisting.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of smallest singular value
        """
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        _, S, _ = np.linalg.svd(G)
        min_sig = S[5]
        return min_sig

    @staticmethod
    def wrench_volume(forces, torques, normals, soft_fingers=False, params=None):
        """ Volume of grasp matrix singular values - score of all wrenches that the grasp can resist.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of wrench volume
        """
        k = 1
        if params is not None and 'k' in params.keys():
            k = params.k

        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        _, S, _ = np.linalg.svd(G)
        sig = S
        return k * np.sqrt(np.prod(sig))

    @staticmethod
    def grasp_isotropy(forces, torques, normals, soft_fingers=False, params=None):
        """ Condition number of grasp matrix - ratio of "weakest" wrench that the grasp can exert to the "strongest" one.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of grasp isotropy metric
        """
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        _, S, _ = np.linalg.svd(G)
        max_sig = S[0]
        min_sig = S[5]
        isotropy = min_sig / max_sig
        if np.isnan(isotropy) or np.isinf(isotropy):
            return 0
        return isotropy

    @staticmethod
    def ferrari_canny_L1(forces, torques, normals, soft_fingers=False, params=None,
                         wrench_norm_thresh=1e-3,
                         wrench_regularizer=1e-10):
      #  print ('gggggggggggggggggggggg')



        """ Ferrari & Canny's L1 metric. Also known as the epsilon metric.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite

        Returns
        -------
        float : value of metric
        """
        if params is not None and 'wrench_norm_thresh' in params.keys():
            wrench_norm_thresh = params.wrench_norm_thresh
        if params is not None and 'wrench_regularizer' in params.keys():
            wrench_regularizer = params.wrench_regularizer
        s_all = time.time()
        # create grasp matrix
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers, params=params)
        s = time.time()
        # center grasp matrix for better convex hull comp
        hull = cvh.ConvexHull(G.T)
        # TODO: suppress ridiculous amount of output for perfectly valid input to qhull
        e = time.time()
        logging.debug('CVH took %.3f sec' %(e - s))
        
        debug = False
        if debug:
            fig = plt.figure()
            torques = G[3:,:].T
            ax = Axes3D(fig)
            ax.scatter(torques[:,0], torques[:,1], torques[:,2], c='b', s=50)
            ax.scatter(0, 0, 0, c='k', s=80)
            ax.set_xlim3d(-1.5, 1.5)
            ax.set_ylim3d(-1.5, 1.5)
            ax.set_zlim3d(-1.5, 1.5)
            ax.set_xlabel('tx')
            ax.set_ylabel('ty')
            ax.set_zlabel('tz')
            plt.savefig("filename"+str(e - s)+".png")
           # plt.show()

        if len(hull.vertices) == 0:
            logging.warning('Convex hull could not be computed')
            return -99
        # determine whether or not zero is in the convex hull
        s = time.time()
        min_norm_in_hull, v = PointGraspMetrics3D.min_norm_vector_in_facet(G, wrench_regularizer=wrench_regularizer)
        #   这个是个难点，用的是cvx优化包来做的。
#求的是方程：
#最小化 0.5 x'Px + q'x subject to Gx <= h, Ax = b
#的x的最优解
#此处P为之前的G^T*G,q趋于0，G为负单位阵，h为0，A为单位阵，b为1

#故本质就是求使得|Gx|最小的x系数（x必须满足凸组合且大于0）
#（为什么要满足满足凸组合？要保证Gx在G的凸包边界上）！！！！！！！！
#所以这个方程的本质含义是什么？就是求在G的凸包边界上距离原点的最小值！！！！！！！！！！！
        e = time.time()
        logging.debug('Min norm took %.3f sec' %(e - s))

        # if norm is greater than 0 then forces are outside of hull  ??????????????这个有问题，应该是在凸包外，这个没有对其进行判断的部分，而是直接设定人为阈值
        minuskey=0
        if min_norm_in_hull > wrench_norm_thresh:
            logging.debug('Zero not in convex hull')
            minuskey=1
            return 0.0

        # if there are fewer nonzeros than D-1 (dim of space minus one)
        # then zero is on the boundary and therefore we do not have
        # force closure
        if np.sum(v > 1e-4) <= G.shape[0]-1:
            logging.debug('Zero not in interior of convex hull')
            return 0.0

        # find minimum norm vector across all facets of convex hull
        s = time.time()
        min_dist = sys.float_info.max
        closest_facet = None
        for v in hull.vertices:
            if np.max(np.array(v)) < G.shape[1]: # because of some occasional odd behavior from pyhull
                facet = G[:, v] #G凸包的G中的角点组成的面片？？
                dist, _ = PointGraspMetrics3D.min_norm_vector_in_facet(facet, wrench_regularizer=wrench_regularizer)#求这个面片到原点的最小值
                if dist < min_dist:
                    min_dist = dist
                    closest_facet = v
        e = time.time()
        logging.debug('Min dist took %.3f sec for %d vertices' %(e - s, len(hull.vertices)))
        e_all = time.time()
#        if minuskey==1:
#           min_dist=-min_dist
        
        return min_dist

    @staticmethod
    def wrench_in_positive_span(wrench_basis, target_wrench, force_limit, num_fingers=1,
                                wrench_norm_thresh = 1e-4, wrench_regularizer = 1e-10):
        """ Check whether a target can be exerted by positive combinations of wrenches in a given basis with L1 norm fonger force limit limit.

        Parameters
        ----------
        wrench_basis : 6xN :obj:`numpy.ndarray`
            basis for the wrench space
        target_wrench : 6x1 :obj:`numpy.ndarray`
            target wrench to resist
        force_limit : float
            L1 upper bound on the forces per finger (aka contact point)
        num_fingers : int
            number of contacts, used to enforce L1 finger constraint
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite

        Returns
        -------
        int
            whether or not wrench can be resisted
        float
            minimum norm of the finger forces required to resist the wrench
        """
        
        
     #   print 'target_wrench:',target_wrench
        
        num_wrenches = wrench_basis.shape[1]

        # quadratic and linear costs
        P = wrench_basis.T.dot(wrench_basis) + wrench_regularizer*np.eye(num_wrenches)
        q = -wrench_basis.T.dot(target_wrench)

        # inequalities
        lam_geq_zero = -1 * np.eye(num_wrenches)
        
        num_wrenches_per_finger = num_wrenches / num_fingers
        force_constraint = np.zeros([num_fingers, num_wrenches])
        for i in range(num_fingers):
            start_i = num_wrenches_per_finger * i
            end_i = num_wrenches_per_finger * (i + 1)
            force_constraint[i, start_i:end_i] = np.ones(num_wrenches_per_finger)

        G = np.r_[lam_geq_zero, force_constraint]
        h = np.zeros(num_wrenches+num_fingers)
        for i in range(num_fingers):
            h[num_wrenches+i] = force_limit

        # convert to cvx and solve
        P = cvx.matrix(P)
        q = cvx.matrix(q)
        G = cvx.matrix(G)
        h = cvx.matrix(h)
        sol = cvx.solvers.qp(P, q, G, h)
        v = np.array(sol['x'])

        min_dist = np.linalg.norm(wrench_basis.dot(v).ravel() - target_wrench)**2

        # add back in the target wrench
        return min_dist < wrench_norm_thresh, np.linalg.norm(v)

    @staticmethod
    def min_norm_vector_in_facet(facet, wrench_regularizer=1e-10):
        """ Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.

        Parameters
        ----------
        facet : 6xN :obj:`numpy.ndarray`
            vectors forming the facet
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite

        Returns
        -------
        float
            minimum norm of any point in the convex hull of the facet
        Nx1 :obj:`numpy.ndarray`
            vector of coefficients that achieves the minimum
        """
        dim = facet.shape[1] # num vertices in facet

        # create alpha weights for vertices of facet
        G = facet.T.dot(facet)
        grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])
        # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b
        P = cvx.matrix(2 * grasp_matrix)   # quadratic cost for Euclidean dist
        q = cvx.matrix(np.zeros((dim, 1)))
        G = cvx.matrix(-np.eye(dim))       # greater than zero constraint
        h = cvx.matrix(np.zeros((dim, 1)))
        A = cvx.matrix(np.ones((1, dim)))  # sum constraint to enforce convex
        b = cvx.matrix(np.ones(1))         # combinations of vertices 这里A和b比较有意思，Ax = b，A为单位阵，b为1，就是要保证x的各个系数和为1
        sol = cvx.solvers.qp(P, q, G, h, A, b)
        v = np.array(sol['x'])
        min_norm = np.sqrt(sol['primal objective'])
        return abs(min_norm), v
