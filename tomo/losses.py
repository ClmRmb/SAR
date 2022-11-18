import numpy as np

def error(points,points_gt):
# Compute the accuracy/error as explained in Rambour et Al.,
# Introducing spatial regularization in SAR tomography reconstruction, TGRS, 2019.
# 
# points = iterable object containing the points in the estimated points clouds
# points_gt = iterable object containing the points in the ground truth
# The points radar coordinates are needed as follow : 
#   pt = [azimuth,range,height]

    points_gt = np.array(points_gt)
    dist = 0
    for pt in points:
        pt = np.array(pt)
        dist += np.min( np.sqrt( np.sum((points_gt - pt)**2,1) ))
    return dist/len(points)

def completeness(points,points_gt):
# Compute the completeness as explained in Rambour et Al.,
# Introducing spatial regularization in SAR tomography reconstruction, TGRS, 2019.
# The completeness is basically the error computed with the estimated points clouds
# as reference
#
# points = iterable object containing the points in the estimated points clouds
# points_gt = iterable object containing the points in the ground truth
# The points radar coordinates are needed as follow : 
#   pt = [azimuth,range,height]

    return error(points_gt,points)