import numpy as np
import cv2
import math as ma


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    fw = feature_width

    assert image.ndim == 2, 'Image must be grayscale'
    feat_dim = 128
    fv = np.ones(feat_dim)
    m,n = np.shape(image)
    w = feature_width//2
    k = x.size
    filt = cv2.getGaussianKernel(ksize=w//2, sigma=6)
    filt = np.dot(filt, filt.T)
    for i in range(0,k):
        xx = np.asscalar(x[i].astype(int))
        yy = np.asscalar(y[i].astype(int))
        if (xx-w<0) or (xx+w >n) or (yy-w<0) or (yy+w > m):
            continue
        temp = np.ones(feat_dim)
        im = image[yy-w:yy+w,xx-w:xx+w]
        im = cv2.filter2D(im,-1,filt)
        im = np.gradient(im)
        orn = np.arctan2(im[0],im[1])+ma.pi
        mag = np.hypot(im[0],im[1])
        mag = mag.reshape(feature_width//4,4,feature_width//4,4).swapaxes(1,2).reshape(-1,feature_width//4,feature_width//4)
        orn = orn.reshape(feature_width//4,4,feature_width//4,4).swapaxes(1,2).reshape(-1,feature_width//4,feature_width//4)
        for o in range(0,16):
            bins = np.zeros(8)
            ornb = orn[o].flatten()
            magb = mag[o].flatten()
            for p in range(ornb.size):
                ind = np.asscalar((np.floor(ornb[p]/(ma.pi/4)).astype(int)))
                if ind==8:
                    ind-=1
                bins[ind] += magb[p]
            temp[(8*o):(8*o+8)] = bins
            temp = temp/np.linalg.norm(temp)
        fv = np.vstack((fv, temp))

    fv = fv[1:,:]
    #print(fv[3])
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv
