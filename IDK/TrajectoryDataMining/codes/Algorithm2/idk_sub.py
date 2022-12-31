import pandas as pd
import numpy as np
from codes.Algorithm1.iNNE_IK import *


def idk_kernel_map(list_of_distributions, psi, t=100):
    D_idx = [0]    # index of each distributions
    alldata = []
    n = len(list_of_distributions)
    for i in range(1, n+1):
        D_idx.append(D_idx[i-1]+len(list_of_distributions[i-1]))
        alldata += list_of_distributions[i-1]
    alldata = np.array(alldata)

    inne_ik = iNN_IK(psi, t)
    all_ikmap = inne_ik.fit_transform(alldata).toarray()

    idkmap = []
    for i in range(n):
        idkmap.append(np.sum(all_ikmap[D_idx[i]:D_idx[i+1]], axis=0) / (D_idx[i+1] - D_idx[i]))
    idkmap = np.array(idkmap)

    return idkmap, all_ikmap


def idk_point(list_of_distributions, psi_1, t_1,psi_2, t_2, ano_traj):
    idk_map1,  all_ikmap= idk_kernel_map(list_of_distributions, psi_1, t_1)
    inne_ik = iNN_IK(psi_2, t_2)
    m = len(list_of_distributions[ano_traj])
    n = 0
    for i in range(ano_traj):
        n+= len(list_of_distributions[i])
    idk_map2,all_ikmap_2 = inne_ik.fit_transform_1(idk_map1,all_ikmap[n:n+m])
    idk_map2 = idk_map2.toarray()
    all_ikmap_2 = all_ikmap_2.toarray()
    idkm2_mean = np.average(idk_map2,axis = 0)
    y=np.zeros(m)
    for i in range(m):
        y[i] = np.dot(all_ikmap_2[i],idkm2_mean.T)/t_2
    return y


def SD_diff(list_of_distributions, psi_1, t_1,psi_2, t_2, ano_traj):
    y = idk_point(list_of_distributions, psi_1, t_1,psi_2, t_2, ano_traj)
    sort_y = np.argsort(y)
    n = len(list_of_distributions[ano_traj])
    t = np.zeros(n)
    for i in range(n):
        t[i] = y[sort_y[i]]
    t_3 = np.zeros(n)
    for i in range(n):
        if i == 0 or i ==n-1:
            t_3[i] = np.var(t)
        else:
            t_3[i] = np.abs(np.var(t[:i])-np.var(t[i:]))
    return sort_y,t,t_3


def find_and_return_position(l1,num):
  for idx,x in enumerate(l1):
    if x==num:
      return idx
  return -1


def subtraj(sort_y,t_3):
    minsize = np.argsort(t_3)
    if 0 == minsize[0]:
        print('No subtrajectories')
        raise Exception
    l_1 = sort_y[:minsize[0]]
    l1_2 = np.sort(l_1).tolist()
    l2_2 = l1_2.copy()

    start_flag = True
    increment_seqs = []
    while len(l1_2)> 0:
      if start_flag:
        nums = [l1_2[0],] #初始化获得的序列
        positions = [0,] #初始化获得序列对应位置
        start_flag = False

      #判断是否存在下个数字，并且顺序递增
      elif  find_and_return_position(l1_2,nums[-1] + 1) > positions[-1] :
        positions.append(find_and_return_position(l1_2,nums[-1] + 1))
        nums.append(nums[-1]+1)

      #失配则保存当前序列，删除当前元素，进入下个循环
      else:

        increment_seqs.append(nums) #保存当前序列
        start_flag = True #设定新序列标记
        #清理已选出序列
        count = 0
        while len(positions)> 0:
          del l1_2[positions[0]-count] #清理序列中元素
          del positions[0] #清理位置标记
          count = count + 1 #每一次删除元素都会造成下标改变，故进行位置调整
        start_flag = True
    start_flag = True
    decrease_seqs = []
    while len(l2_2)> 0:
      if start_flag:
        nums = [l2_2[0],] #初始化获得的序列
        positions = [0,] #初始化获得序列对应位置
        start_flag = False

      #判断是否存在下个数字，并且顺序递增
      elif  len(positions)>0 and find_and_return_position(l2_2,nums[-1] - 1) > positions[-1]  :
        positions.append(find_and_return_position(l2_2,nums[-1] - 1))
        nums.append(nums[-1]-1)

      #失配则保存当前序列，删除当前元素，进入下个循环
      else:
        decrease_seqs.append(nums) #保存当前序列
        start_flag = True #设定新序列标记
        #清理已选出序列
        count = 0
        while len(positions)> 0:
          del l2_2[positions[0]-count] #清理序列中元素
          del positions[0] #清理位置标记
          count = count + 1 #每一次删除元素都会造成下标改变，故进行位置调整
        start_flag = True
    increment_result = []
    for x in increment_seqs:
      if len(x)>1:
        increment_result.append(x)
    return  increment_result