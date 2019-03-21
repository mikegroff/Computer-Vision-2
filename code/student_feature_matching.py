import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    th = 0.99
    k, len1 = np.shape(features1)
    j, len2 = np.shape(features2)
    match = np.ones(2)
    cons = np.ones(1)
    r = np.power((np.max(x2)- np.min(x2))*(np.max(y2)- np.min(y2))*(np.max(x1)- np.min(x1))*(np.max(y1)- np.min(y1)),0.25)*0.15
    for i in range(0,k):
        diff = np.zeros(j)
        for l in range(0,j):
            diff[l] = np.linalg.norm(features1[i]-features2[l])
        sortd = np.sort(diff)
        sorta = np.argsort(diff)
        v = sorta[0]
        p = v
        dist = np.sqrt(np.square(x1[i]-x2[v])+ np.square(y1[i]-y2[v]))
        s = 0
        while(dist > r) and (s < j-2):
            s += 1
            v = sorta[s]
            dist = np.sqrt(np.square(x1[i]-x2[v])+ np.square(y1[i]-y2[v]))
        nndr = sortd[0+s]/sortd[1+s]
        #print(s)
        if (nndr < th ) and (s < j/10):
            match = np.vstack((match,np.array([i,v])))
            cons = np.append(cons,np.array([nndr]))
    match = match[1:,:]
    cons = cons[1:]
    #print(match)
        #check for matches in opposite direction

    confidences = cons
    consort = np.argsort(confidences)
    confidences = np.take(confidences, consort)

    matches = match
    #print(matches)
    matches[:,0] = np.take(matches[:,0], consort)
    matches[:,1] = np.take(matches[:,1], consort)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches.astype(int), confidences
