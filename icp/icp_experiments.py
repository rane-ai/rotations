import os
import copy
import time
import numpy as np
import open3d as o3d
from scipy.linalg import solve
import matplotlib.pyplot as plt
from numba_kdtree import KDTree
from numba import jit, prange

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

'''
A simple implementation of ICP demonstrating different rigid body parametersations. The code has been written 
for clarity/ease-of-understanding and is definitely not optimal! The point to point distance between 2 (roughly 
aligned) scans is minimised using Gauss Newton. The approach described in [0] was used to optimise the relative 
pose. Three parameterisations for rigid body increments delta_X are considered:

(1) Euler :
        delta_X = [delta_x, delta_y, delta_z, delta_phi, delta_gamma, delta_psi]

(2) Twist (linear/angular velocities):
        delta_X = [u_x, u_y, u_z, omega_x, omega_y, omega_z]

(3) Quaternion:
        delta_X = [delta_x, delta_y, delta_z, q_x, q_y, q_z]

There is a lot of room for improvement e.g. point to plane metric, solver etc.

[0]  "Least Squares Optimization: from Theory to Practice" by Grisetti et al https://arxiv.org/pdf/2002.11051.pdf

'''

methods = ['euclidean', 'manifold', 'quaternion']


def draw_registration_result(source, target, target_T_source):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([245/255.0, 66/255.0, 164/255.0])
    target_temp.paint_uniform_color([66/255.0, 245/255.0, 66/255.0])
    target_temp.transform( np.linalg.inv( target_T_source )  )
    o3d.visualization.draw_geometries([source_temp, target_temp])


def animate_reg(source, 
                target, 
                delta_xs):
    '''
    Shows live view of optimisation pulling point clouds together
    '''
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)
    for delta_x in delta_xs:
        target.transform( np.linalg.inv(v2t_euc(delta_x)) )
        vis.update_geometry(target)
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

    draw_registration_result(source, target, np.identity(4))


@jit(nopython=True)
def get_rotation_matrix(phi, gamma, psi):

    roll, pitch, yaw = phi, gamma, psi

    R_x = np.identity(4)
    R_y = np.identity(4)
    R_z = np.identity(4)
    
    R_x[1, 1] =  np.cos(roll)
    R_x[1, 2] = -np.sin(roll)
    R_x[2, 1] =  np.sin(roll)
    R_x[2, 2]=   np.cos(roll)

    R_y[0, 0] =  np.cos(pitch)
    R_y[0, 2] =  np.sin(pitch)
    R_y[2, 0] = -np.sin(pitch) 
    R_y[2, 2] =  np.cos(pitch)

    R_z[0, 0] =  np.cos(yaw)
    R_z[0, 1] = -np.sin(yaw)
    R_z[1, 0] =  np.sin(yaw)
    R_z[1, 1] =  np.cos(yaw)
    
    R = np.dot(np.dot(R_x, R_y), R_z)
    return R


@jit(nopython=True)
def to_skew(a):

    a_1, a_2, a_3 = a[0], a[1], a[2]

    a_skew = np.array([[   0,  -a_3,  a_2],
                       [ a_3,     0, -a_1],
                       [-a_2,   a_1,    0]])
    return a_skew


@jit(nopython=True)
def vee(a):
    a_1 = a[2,1]
    a_2 = a[0,2]
    a_3 = a[1,0]
    return np.array([a_1, a_2, a_3])


#------------------------------------
# EUCLIDEAN
#------------------------------------
@jit(nopython=True)
def J_icp_euc(T, p):
    '''
    Derivative of the observation function w.r.t the Euclidean increment
    '''
    R = T[0:3,0:3]
    res = np.dot( -R.transpose(), np.hstack( (np.eye(3), -to_skew(p)) ) )
    return res  


@jit(nopython=True)
def t2v_euc(T):
    x, y, z = T[0,3], T[1,3], T[2,3]
    r = T[0:3, 0:3]
    phi   = np.arctan2( -r[1,2],  r[2,2] )
    psi   = np.arctan2( -r[0,1],  r[0,0])
    gamma = np.arctan2(  r[0,2], (r[0,0]/np.cos(psi)) )
    x_vec = np.array([ x, y, z, phi, gamma, psi ])
    return x_vec


@jit(nopython=True)
def v2t_euc(x_vec):
    assert x_vec.ndim == 1
    x, y, z, phi, gamma, psi = x_vec
    T = get_rotation_matrix(phi, gamma, psi)
    T[0:3,3] = np.array([x,y,z])
    return T
#------------------------------------


#------------------------------------
# MANIFOLD
#------------------------------------
@jit(nopython=True)
def t2v_lie(T):
    '''
    Maps Manifold to Lie algebra
    '''

    t = T[0:3,3]
    R = T[0:3, 0:3]

    theta = np.arccos( (np.trace(R) - 1)/2.0 )

    ln_R = (theta / (2*np.sin(theta)) ) * (R - R.transpose())
    
    omega = vee(ln_R)

    omega_skew = to_skew(omega)
    omega_skew_sq = np.dot( omega_skew, omega_skew )

    A = np.sin(theta)/theta
    B = (1 - np.cos(theta))/( theta*theta )
    theta_sq = theta**2

    V_inv = np.eye(3) - 0.5*omega_skew + (1/theta_sq)*  (1 - (A/(2*B))) * omega_skew_sq 

    u = np.dot( V_inv, t)

    u_omega = np.array([ u[0], u[1], u[2], omega[0], omega[1], omega[2] ])

    return u_omega


@jit(nopython=True)
def v2t_lie(u_omega):
    '''
    Maps from lie algebra (tangent space) to Manifold
    '''
    assert u_omega.ndim == 1

    u     = u_omega[0:3]
    omega = u_omega[3:6]

    theta = np.linalg.norm(omega)

    A = np.sin(theta)/theta
    B = (1 - np.cos(theta))/( theta*theta )
    C = (1 - A)/(theta*theta)

    omega_skew = to_skew(omega)

    omega_skew_sq = np.dot( omega_skew, omega_skew )
    R = np.eye(3) + A*omega_skew + B*omega_skew_sq
    V = np.eye(3) + B*omega_skew + C*omega_skew_sq

    t = np.dot(V,u)

    # assert (np.linalg.det(R) - 1.0) < 1e-6, np.linalg.det(R)

    T = np.identity(4)
    T[0:3,0:3] = R
    T[0:3,3]   = np.expand_dims(t, axis=0)
    return T


@jit(nopython=True)
def J_icp_lie(T, p):
    '''
    Derivative of the observation function w.r.t the twist
    '''
    R = T[0:3,0:3]
    res = np.dot( -R.transpose(), np.hstack( (np.eye(3), -to_skew(p)) ) )
    return res
#------------------------------------


#------------------------------------
# QUATERNION
#------------------------------------
@jit(nopython=True)
def J_icp_q(T, p):
    '''
    Derivative of the observation function w.r.t the quaternion based state increment
    '''
    R = T[0:3,0:3]
    p_x, p_y, p_z = p[0], p[1], p[2]
    X = np.array([[ -1,  0,  0,  0,     -2*p_z,  2*p_y ],
                  [  0, -1,  0,  2*p_z,  0,     -2*p_x ],
                  [  0,  0, -1, -2*p_y,  2*p_x,  0     ]])

    res = np.dot( R.transpose(), X )
    return res


@jit(nopython=True)
def t2v_q(T):

    R = T[0:3,0:3]

    e = np.linalg.norm( R.flatten() - np.eye(3).flatten() )
    if e < 1e-9:
        return np.array([ T[0,3], T[1,3], T[2,3], 0, 0, 0, 1 ])

    theta = np.arccos( (np.trace(R) - 1)/2.0 )

    ln_R = (theta / (2*np.sin(theta)) ) * (R - R.transpose())

    omega = vee(ln_R)
    u = omega/theta

    u_x, u_y, u_z = u

    q_w =     np.cos(theta/2.0)
    q_x = u_x*np.sin(theta/2.0)
    q_y = u_y*np.sin(theta/2.0)
    q_z = u_z*np.sin(theta/2.0)

    q = np.array([ T[0,3], T[1,3], T[2,3], q_x, q_y, q_z ])
    return q


@jit(nopython=True)
def v2t_q(x_vec):
    '''
    Note the real value of the quaternion is recovered from the unit magnitude constraint of quaternions that represent
    rotations
    '''
    x, y, z, q_x, q_y, q_z = x_vec

    q_w = np.sqrt( 1 - (q_x**2 + q_y**2 + q_z**2) )
 
    R = np.array([[(q_w*q_w + q_x*q_x - q_y*q_y - q_z*q_z), 2*(q_x*q_y-q_w*q_z),             2*(q_x*q_z+q_w*q_y)            ],
                  [2*(q_x*q_y+q_w*q_z),            (q_w*q_w - q_x*q_x + q_y*q_y - q_z*q_z), 2*(q_y*q_z-q_w*q_x)            ],
                  [2*(q_x*q_z-q_w*q_y),            2*(q_y*q_z+q_w*q_x),             (q_w*q_w - q_x*q_x - q_y*q_y + q_z*q_z)]])
    
    T = np.identity(4)
    T[0:3,0:3] = R
    T[0:3,3]   = np.array([x,y,z])
    return T
#------------------------------------


@jit(nopython=True)
def h_icp(target_T_source, target_m):
    '''
    Transforms moving point cloud into the frame of the static point cloud.
    Error can the be evaluated in a common frame.
    '''
    
    assert target_T_source.shape == (4,4)

    N = target_m.shape[0]
    p = np.hstack((target_m, np.ones((N,1))) )

    source_m = np.dot( np.linalg.inv(target_T_source), p.transpose() ).transpose()

    return source_m[:,0:3]
    

@jit(nopython=True)
def core(x_vec, source_tree, source_pcd_points, target_m, omega_hat, method):
    '''
    Builds matrix, H, and vector, b, which are used when solving for the state
    increment. 
    '''

    if method == 'euclidean':
        source_m = h_icp(v2t_euc(x_vec), target_m)
    elif method == 'manifold':
        source_m = h_icp(v2t_lie(x_vec), target_m)
    elif method == 'quaternion':
        source_m = h_icp(v2t_q(x_vec), target_m)
    else:
        raise Exception("Unknown method!")
    
    F_curr = 0
    H = np.zeros((6,6))
    b = np.zeros((6,1))
    for point_idx in prange(source_m.shape[0]):
        
        # For each source point find closest in the target:
        #   - Z_hat is the prediction
        #   - Z     is the measurement

        Z_hat = source_m[point_idx,:]    
        _, idx = source_tree.query(Z_hat, k=1)
        idx = idx[0][0]

        Z = source_pcd_points[idx,:]
          
        # Compute error between the prediction (moved target point) and measurement (static source point)
        e_k  = np.expand_dims( Z_hat - Z, axis = 1) # e is 3x1

        chi_k   = np.dot(e_k.transpose(), np.dot(omega_hat, e_k))[0][0] # chi_k is a scalar
        F_curr += chi_k

        # Robustifying function
        u_k     = np.sqrt(chi_k)
        gamma_k = 1.0/u_k
        sigma_k_tilde = gamma_k*omega_hat

        # Compute terms in equation to be solved
        if method == 'euclidean':
            J = J_icp_euc( v2t_euc(x_vec) , Z_hat) # 3 x 6
        elif method == 'manifold':
            J = J_icp_lie( v2t_lie(x_vec) , Z_hat) # 3 x 6
        elif method == 'quaternion':
            J = J_icp_q( v2t_q(x_vec) , Z_hat) # 3 x 6

        H += np.dot( J.transpose(), np.dot( sigma_k_tilde, J  )) # 6 x 6
        b += np.dot( J.transpose(), np.dot( sigma_k_tilde, e_k)) # 6 x 1

    return H, b, F_curr


def gn_min(source_pcd, # Static point cloud
           target_pcd, # Moving point cloud
           x_vec_init, # Initial relative pose (different parameterisations)
           loss_val,
           x_delta_list,
           method,
           epsilon  = 1e-7,
           max_iter = 250): 
    '''
    Gauss Newton Minimisation. Minimise point-point distance between point clouds 
    by optimising the relative pose. This has been written for easy understanding 
    rather than speed!
    '''

    F_prev, F_curr = np.inf, 0 # Scalar Loss

    point_cov = np.eye(3)*np.sqrt( 0.02 ) # Covariance associated with a point
    omega_hat = np.linalg.inv(point_cov)

    print("Covariance (m)")
    print(point_cov)
    print("")
    target_m = np.asarray( target_pcd.points )
   
    source_tree = KDTree( np.asarray( source_pcd.points ), leafsize=10)

    source_pcd_points = np.asarray( source_pcd.points )

    ctr = 0
    x_vec = x_vec_init
    prev_x_vec = np.array([0,0,0,0,0,0])
    start_time = time.time()
    while (np.abs(F_prev - F_curr) > epsilon) and (ctr < max_iter):
        
        F_prev = F_curr
        H, b, F_curr = core(x_vec, source_tree, source_pcd_points, target_m, omega_hat, method )
     
        delta_x_vec = solve(H, -b).squeeze()  

        x_delta_list.append( delta_x_vec )

        loss_val.append(F_curr)
        displacement_mm = np.linalg.norm(delta_x_vec[0:3])*1000 
        print("Iteration {} | Loss Diff = {:.7f} | Displacement = {:.4f}mm".format( ctr, np.abs(F_prev - F_curr), displacement_mm  ))
        prev_x_vec = delta_x_vec
        x_vec += delta_x_vec
        ctr+=1

    T_final = v2t_euc(x_vec) 

    print("No. of iterations {}".format( ctr ))
    print("Took {:.3f}secs".format( time.time()-start_time ))

    return T_final, x_delta_list

   
def load_and_transform_point_clouds():

    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source_raw    = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    target_raw    = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

    source = source_raw.voxel_down_sample(voxel_size=0.02)
    target = target_raw.voxel_down_sample(voxel_size=0.02)
        
    # Make them close enough...
    trans = np.array([[ 0.862, 0.011, -0.507,  0.0+0.25], 
                        [-0.139, 0.967, -0.215,  0.7-.1],
                        [ 0.487, 0.255,  0.835, -1.4], 
                        [ 0.0,   0.0,    0.0,    1.0]])

    source.transform(trans)

    T = get_rotation_matrix(5*np.pi/180, 3*np.pi/180, 5*np.pi/180)
    source.transform(T)
        
    return source, target


# http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
def register_point_cloud_pair(method, viz=True):
    
    print("Method = {}".format( method ))

    source, target = load_and_transform_point_clouds()
    target_orig = copy.deepcopy(target)
 
    if viz:
        draw_registration_result(source, target, np.identity(4))
    
    # Uncomment to playback the reg
    # animate_reg(source, target, np.load('deltas.npy') ),exit()

    x_vec_init = np.array([0.0, 0.0, 0.0, 1e-6, 1e-6, 1e-6])

    loss_val = []
    x_delta_list = []
    T, x_delta_list = gn_min(source, target, x_vec_init, loss_val, x_delta_list, method)
    np.save("./deltas.npy", x_delta_list)

    if viz:
        animate_reg(source, target_orig, x_delta_list)

    print(T)
    print(np.linalg.inv(T))

    if viz:
        plt.figure()
        plt.plot(loss_val)
        plt.xlabel('Iteration')
        plt.ylabel("Loss")
        plt.title("Loss Function | min = {:.3f}".format( np.min(loss_val) ))
        plt.grid()
        plt.show()

    return loss_val
        

def compare_methods():
    plt.figure()
    for method in methods:
        print(method)
        loss = register_point_cloud_pair(method, viz=False)
        plt.plot(loss, label=method)
    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":

    method = methods[0]
    register_point_cloud_pair(method, viz=True)



