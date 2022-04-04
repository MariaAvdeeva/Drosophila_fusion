## from https://github.com/ClayFlannigan/icp/blob/master/icp.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def scale(A, c, m):
    sc = A[:m,:].T.copy()
    centroid = np.mean(sc, axis=0)
    centered = (sc - centroid)
    scaled = A.copy()
    scaled[:m,:] = (centered*c+centroid).T
    return scaled

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst, check_shape = True, distance_matrix = None):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    if check_shape:
        assert src.shape == dst.shape

    if distance_matrix is None:
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
    else:
        indices = np.argmin(distance_matrix,1)
        distances = np.array([distance_matrix[i, indices[i]] for i in range(src.shape[0])])
        print(distances)
    return distances.ravel(), indices.ravel()


def icp_nn(A, B, init_pose=None, max_iterations=20, tolerance=0.001, scaling = False, plot = False, distance_matrix = None):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    final_scale = 1
    opt_means = []    
    if scaling:
        for i in range(max_iterations):
            # compute the transformation between the current source and nearest destination points
            opt_mean = np.inf
            if (i>1):
                for c in np.arange(0.8, 1.2, 0.01):
                    src_c = scale(src, c, m)
                    # find the nearest neighbors between the current source and destination points
                    distances, indices = nearest_neighbor(src_c[:m,:].T, dst[:m,:].T, distance_matrix = distance_matrix)
                    cur_mean = np.mean(distances)
                    print(c, cur_mean)
                    if cur_mean < opt_mean:
                        T,_,_ = best_fit_transform(src_c[:m,:].T, dst[:m,indices].T)
                        opt_c = c
                        opt_src = src_c
                        opt_mean = cur_mean
                        opt_distances = distances  
                opt_means.append(opt_mean)
                #print(opt_c)
                src = np.dot(T, opt_src)
                final_scale*=opt_c
                # check error
                mean_error = opt_mean
            else:
                distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T, distance_matrix = distance_matrix)
                T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
                src = np.dot(T, src)
                mean_error = np.mean(distances)
                opt_means.append(mean_error)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error
    else:
        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T, distance_matrix = distance_matrix)
            mean_error = np.mean(distances)
            #print(mean_error)
            opt_means.append(mean_error)
            # compute the transformation between the current source and nearest destination points
            T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error
    if plot:
        plt.plot(opt_means)

    opt_means.append(mean_error)
    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)
    transformed= src[:m,:]

    return T, transformed, final_scale, distances, i, mean_error, opt_means