import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time
import scipy.signal as sp


#code to profile


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    t1=time()
    confidences, scales, orientations = None, None, None
    m,n = np.shape(image)
    scales = np.array([m,n])
    xx = np.zeros(1)
    yy = np.zeros(1)
    cons = np.zeros(1)
    w = feature_width//4
    th = 1000
    feat = image
    R = np.zeros((m,n))
    filt = cv2.getGaussianKernel(ksize=w//2, sigma=1)
    filt = np.dot(filt, filt.T)
    sobelx = cv2.Sobel(feat,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(feat,cv2.CV_64F,0,1,ksize=5)
    Sx2 = cv2.filter2D(sobelx*sobelx,-1,filt)
    Sy2 = cv2.filter2D(sobely*sobely,-1,filt)
    Sxy = cv2.filter2D(sobelx*sobely,-1,filt)
    det = Sx2*Sy2 - Sxy**2
    trac = Sx2+Sxy
    R = det-0.06*(trac**2)
    #ANMS
    w = 1
    for h in range(feature_width,m-feature_width):
        for j in range(feature_width,n-feature_width):
            y = h
            x = j
            v = (R[h,j])
            if v < th:
                continue
            feat = R[y-w:y+w,x-w:x+w].flatten()
            sortd = np.sort(feat)
            next = sortd[-2]
            if (0.9*v > next):
                xx = np.append(xx, np.array([x]))
                yy = np.append(yy, np.array([y]))
                cons = np.append(cons, np.array([v/next]))
    xx = xx[1:]
    yy = yy[1:]
    confidences = cons[1:]
    consort = np.argsort(confidences)
    if consort.size>10000:
        consort = consort[0:10000]
    confidences = np.take(confidences, consort)
    xx = np.take(xx,consort)
    yy = np.take(yy,consort)
    t2=time()
    Runtime=t2-t1
    #print(Runtime)

    return xx,yy,confidences, scales, orientations
