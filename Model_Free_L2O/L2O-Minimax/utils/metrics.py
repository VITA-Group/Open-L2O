
import numpy as np
import math
def is_converge(v,k1=10000,k2=1000, threshold1=0.08,threshold2=0.01):
    flag1 = 1
    temp1 = v[-1*k1:]
    for t in temp1:
        if abs(t) > threshold1:
            flag1 = 0
    flag2 = 1
    temp2 = v[-1*k2:]
    for t in temp2:
        if abs(t) > threshold2:
            flag2 = 0
    return flag1 or flag2


def get_index(all_distance_ever1,all_distance_ever2=None,iteration_index=-1,unroll_unit=-1, batch_data=None, ratio=0.8, option=7):
    result = []
    k = int(ratio * len(all_distance_ever1))
    curve_info = all_distance_ever1
    flag = iteration_index//10

    for i in range(len(all_distance_ever1)):
        a = batch_data[i, 0]
        b = batch_data[i, 1]

    curve_v = curve_info[:, 0:flag * 2 * unroll_unit - 1, 1] # -1 here indicates that the curve info at 10th iteration is not recorded
    curve_u = curve_info[:, 0:flag * 2 * unroll_unit - 1, 0]  # curve information so far

    for i in range(len(curve_v)):
        a = batch_data[i,0]
        b = batch_data[i,1]
        grad_v = [(-1) * b * math.sin(a * math.pi * u) for u in curve_u[i]]
        grad_u = [(-1) * b * curve_v[i][j] * math.cos(a * math.pi * curve_u[i][j]) * a * math.pi for j in range(len(curve_v[i]))]
        grad_v_squared = [d ** 2 for d in grad_v]
        grad_u_squared = [d ** 2 for d in grad_u]
        result.append(sum(grad_v_squared))

    return sorted(range(len(result)), key=lambda i: result[i], reverse=True)[-k:]
    # get the index of K elements with best performance(least v variance)