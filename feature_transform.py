import math
import numpy as np


def calc_mean(data):
    mean = [0, 0]
    for x in data:
        mean[0] += x[0]
        mean[1] += x[1]
    mean[0] /= len(data)
    mean[1] /= len(data)
    return mean


def feature_transform(l, r, fixed_scale=False):
    '''
        Calculate the feature transform from l to r.

        Returns the calculated:
        * scale
        * rotation
        * transition in x
        * transition in y
        * center of transformation
    '''
    if len(r) != len(l):
        print("Invalid dataset, l and r must have the same size!")
        return None, None, None, None, None
    l_mean = np.array(calc_mean(l))
    r_mean = np.array(calc_mean(r))

    cs = 0
    ss = 0
    rr = 0
    ll = 0
    for i in range(len(l)):
        l_i = np.array(l[i]) - l_mean
        r_i = np.array(r[i]) - r_mean

        cs += r_i[0] * l_i[0] + r_i[1] * l_i[1]
        ss += -r_i[0] * l_i[1] + r_i[1] * l_i[0]
        rr += r_i[0] * r_i[0] + r_i[1] * r_i[1]
        ll += l_i[0] * l_i[0] + l_i[1] * l_i[1]

    scale = 1

    if not fixed_scale:
        scale = math.sqrt(rr/ll)

    denom = math.sqrt(cs * cs + ss * ss)
    if abs(denom) < 0.00001:
        print("Numerical instability, denominator is: " + str(denom))
        return 1, 0, 0, 0, l_mean
    c = cs / denom
    s = ss / denom
    theta = math.atan2(s, c)
    tx = r_mean[0] - scale * (c * l_mean[0] - s * l_mean[1])
    ty = r_mean[1] - scale * (s * l_mean[0] + c * l_mean[1])

    return scale, theta, tx, ty, l_mean
