from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import itertools


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START CODE HERE ***
    
    #raise NotImplementedError('init_centroids function not implemented')

    centroids_init= np.zeros((num_clusters, image.shape[-1]))
    unique_pairs = [(x, y) for x in range(0, image.shape[0]) for y in range(0, image.shape[1])]
    sampled_points = random.sample(unique_pairs, num_clusters)
    for i in range(0, len(sampled_points)):
        centroids_init[i] = image[sampled_points[i][0]][sampled_points[i][1]]

    # *** END CODE HERE ***
    print(centroids_init)
    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """
    new_centroids = centroids
    assigned_centroids = np.zeros((image.shape[0], image.shape[1], 1))
    # *** START CODE HERE ***
    #raise NotImplementedError('update_centroids function not implemented')
        # Usually expected to converge long before `max_iter` iterations
                # Initialize `dist` vector to keep track of distance to every centroid
                # Loop over all centroids and store distances in `dist`
                # Find closest centroid and update `new_centroids`
        # Update `new_centroids`
    for i in range(max_iter):
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                pixel = image[x][y]
                c_distances = []
                for c in range(len(new_centroids)):
                    centroid = new_centroids[c]
                    c_distances.append(np.linalg.norm(pixel-centroid))
                assigned_cluster_pixel = c_distances.index(min(c_distances))
                assigned_centroids[x][y] = assigned_cluster_pixel
                
        for c in range(len(new_centroids)):
            assigned_pixels = [image[x][y] for x in range(0, image.shape[0]) for y in range(0, image.shape[1]) if assigned_centroids[x][y]==c]
            if len(assigned_pixels)>0:
                new_centroids[c] = [np.mean([z[0] for z in assigned_pixels]),
                                    np.mean([z[1] for z in assigned_pixels]),
                                    np.mean([z[2] for z in assigned_pixels])]
    # *** END CODE HERE ***
    print(new_centroids)
    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START CODE HERE ***
    #raise NotImplementedError('update_image function not implemented')
            # Initialize `dist` vector to keep track of distance to every centroid
            # Loop over all centroids and store distances in `dist`
            # Find closest centroid and update pixel value in `image`
    # *** END CODE HERE ***
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pixel = image[x][y]
            dist_p = {}
            for c in centroids:
                dist_p.update({np.linalg.norm(pixel-c):c})
            centroid_matched = dist_p[min(dist_p)]
            #print(centroid_matched)
            image[x][y]=centroid_matched
    print(image)
    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, 10, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=30,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
