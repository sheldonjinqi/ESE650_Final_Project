import numpy as np
import random
from collections import defaultdict
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from tqdm import tqdm
import scipy

# initialization
n_max = 85000  # 85000 works
sensing_range = 2
prior_mean = np.array([6, 6])
prior_mean_1 = np.array([6, 6])
prior_cov_1 = np.array([[0.25, 0], [0, 0.25]])
prior_mean_2 = np.array([8, 9])
prior_cov_2 = np.array([[0.25, 0], [0, 0.25]])
prior_mean_3 = np.array([5, 2])
prior_cov_3 = np.array([[0.25, 0], [0, 0.25]])
prior_mean_4 = np.array([3, 7])
prior_cov_4 = np.array([[0.25, 0], [0, 0.25]])
prior_cov = np.array([[0.25, 0], [0, 0.25]])
p_init = np.array([0.0, 0.0])
# q_init = [p_init, prior_cov, np.linalg.det(prior_cov), 0, 0, 0]
# V = [q_init]

# initial_det = np.linalg.det(prior_cov)
# cost_list = [initial_det]
step_size = 0.1
u_all = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]) * step_size
obstacle_space = [[2, 8], [35, 15], [5, 65], [95, 2]]


# construct a few targets
# initialize with a few robots
targets = np.array([[6,6], [8,9], [5,2], [3,7]])
robots = np.array([[0,0]])

obstacle = np.array([[[2,8],[3,9]], [[4,1],[5,3]]])
class node(dict):
    def __init__(self, position, cov, parent):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        self.p = position
        self.Sigma = cov
        self.parent = parent

class Vertex:
    # collection of all nodes
    def __init__(self):
        self.nodes = []
        self.group_map = {} # keeps the map from p to the group id
        self.group = []  # each item is node id that belongs to the same group
        self.max_depth = -1
        self.group_num = 0
        self.depthDict = defaultdict(list) # key: depth, value: node_id
        # from node_id we know its p and thus its group id

    def add(self, p, Sigma, parent):
        self.nodes.append(node(p, Sigma, parent))
        node_id = len(self.nodes) - 1
        if parent is None:
            self.nodes[node_id].depth = 0
        else:
            self.nodes[node_id].depth = self.nodes[parent].depth + 1
        self.depthDict[self.nodes[node_id].depth].append(node_id)
        if self.nodes[node_id].depth > self.max_depth:
            self.max_depth = self.nodes[node_id].depth
        key = (p[0], p[1])
        if key in self.group_map:
            self.group[self.group_map[key]].append(node_id)
        else:
            self.group_map[key] = self.group_num
            self.group_num += 1
            self.group.append([node_id])
        return node_id

def line_rect(p11, p12, rect):
    '''
    rect 4 vertices
    0    1

    2    3
    '''
    return line_line(p11, p12, rect[0], rect[1]) or \
           line_line(p11, p12, rect[0], rect[2]) or \
           line_line(p11, p12, rect[1], rect[3]) or \
           line_line(p11, p12, rect[2], rect[3])
    
def line_line(p11, p12, p21, p22):
    '''
    given two points p11, p12 of line segment 1 and two points p21, p22 of line segment 2 

    reference: http://www.jeffreythompson.org/collision-detection/line-line.php
    '''
    uA = ((p22[0]-p21[0]) * (p11[1]-p21[1]) - (p22[1]-p21[1]) * (p11[0]-p21[0])) / ((p22[1]-p21[1]) * (p12[0]-p11[0]) - (p22[0]-p21[0]) * (p12[1]-p11[1]))
    uB = ((p12[0]-p11[0]) * (p11[1]-p21[1]) - (p12[1]-p11[1]) * (p11[0]-p21[0])) / ((p22[1]-p21[1]) * (p12[0]-p11[0]) - (p22[0]-p21[0]) * (p12[1]-p11[1]))
    return 0 <= uA <= 1 and 0 <= uB <= 1

def is_free(q, obstacles):
    return not np.logical_and(np.all(q >= obstacles[:, 0, :],axis=1), 
                        np.all(q <= obstacles[:, 1, :],axis=1)).any()


def traceback(current_node_id, V):
    path_list = []
    while V[current_node_id].parent != None:
        path_list.append(V[current_node_id].p)
        current_node_id = V[current_node_id].parent
    path_list.append(V[current_node_id].p)
    path_list.reverse()
    return path_list

def RiccatiMap(p, Sigma, target):
    '''
    assume the dynamics is 
    x_{t+1} = Ax_t + w_t, y_t = Mx_t + v_t
    output Sigma is the solution to the Riccati recursion
    A Sigma A.T + W - A Sigma M.T (M Sigma M.T + V)^-1 C Sigma A.T 
    '''
    # M is linearized at p
    geo_dis = np.sqrt(np.square(p - target).sum())
    if geo_dis == 0:
        M = np.zeros((1, 2))
    else:
        M = (p - target) / geo_dis
    return A @ Sigma @ A.T + W - A @ Sigma @ M.T @ np.linalg.inv(M @ Sigma @ M.T + V)@ M @ Sigma @ A.T 

def sample_fv(V, pv=.7):
    ''' 
    among all vertex in V, those with longest path share weight pv and others share 1 - pv
    '''
    # compute group id where nodes with max depth isV.nodes[node_id].p
    K_max = [V.group_map[(V.nodes[node_id].p[0], V.nodes[node_id].p[1])] for node_id in V.depthDict[V.max_depth]]
    all_group = np.arange(len(V.group))
    if len(all_group) - len(K_max) == 0:
        prob = np.ones_like(all_group) / len(K_max)
    else:
        prob = np.zeros_like(all_group) + (1 - pv) / (len(all_group) - len(K_max))
        prob[K_max] = pv / len(K_max)
    return np.random.choice(all_group, p=prob)

def distance_robot_target(j, i):
    '''
    compute the distance between robot j and target i
    '''
    return np.linalg.norm(targets[j] - robots[i])

def distance_robot_x(j, i):
    return np.linalg.norm(targets[j] - x[i])

def target_assign(assignment, satisfied):
    '''
    N robot to M target assignment (N < M) 
    assignment: N by M 0,1 matrix
    satisfied: M by 1 
    '''
    N, M = assignment.shape
    # construct D: those who need to change their target
    # case1: det Sigma_i <= delta_i
    # case2: more than one robot responsible for the same target
    overlap = assignment.sum(axis=0) > 1
    col2update = np.logical_not(np.logical_or(satisfied, overlap))
    D = assignment[col2update].sum(axis=1) > 0
    # T_to_assign = unsatisfied - assigned
    assigned = assignment.sum(axis=0) > 0
    T2assign = np.logical_not(np.logical_or(assigned, satisfied))
    # if all assigned then we reassign
    if T2assign.sum() == 0:
        T2assign = np.logical_not(satisfied)
    for j in D:
        target_idx = np.arange(M)[T2assign][0]
        closest = np.argmin(distance_robot_target(j, target_idx))
        i_closest = target_idx[closest]
        assignment[j] = 0
        assignment[j, i_closet] = 1
        T_assign[closest] = False
        # if all assigned then we reassign
        if T2assign.sum() == 0:
            T2assign = np.logical_not(satisfied)

def get_target(assignment, j):
    N, M = assignment.shape
    return np.arange(M)[assignment[j]][0]

def sample_fu(assignment, Range=2, pu=0.5):
    '''
    u biased towards the one that makes robot j next position p_j
    close to predicted position x_hat_i
    '''
    N, M = assignment.shape
    dis = []
    for j in range(N):
        p_j_next = [dynamics(robots[j], u) for u in u_all]
        i = get_target(assignment, j)
        dis.append(distance_robot_x(j, i))
    u_star = np.argmin(dis)
    if dis[u_star] > Range:
        prob = np.zeros(u_all.shape[0]) + (1 - pu) / (len(u_all) - 1)
        prob[u_star] = pu
    else:
        prob = np.zeros(u_all.shape[0]) + 1 / len(u_all)
    return np.random.choice(np.arange(len(u_all)), p=prob)
    
def dynamics(p, u):
    return p + u

def main_loop(n_max, hidden_Sigma, p0):
    nodes_good = []
    V = Vertex()
    for p in p0:
        V.add(p, hidden_Sigma, None)
    thresh = 1e-5
    for n in range(n_max):
        v_k_rand = sample_fv(V)
        # all q_rand in V.group[v_k_rand] has the same p
        p_rand_id = V.group[v_k_rand][0]
        p_rand = V.nodes[p_rand_id].p
        u_new = u_all[sample_fu(assignment)]
        p_new = dynamics(p_rand, u_new)

        if is_free(p_new, obstacle):
            # taking all possible 
            for q_rand_id in V.group[v_k_rand]:
                target = assignment[j]
                Sigma_new = RiccatiMap(p_rand, V.nodes[q_rand_id].Sigma, target)
                q_new_id = V.add(p_new, Sigma_new, q_rand_id)
                # V[q_new_id].parent = q_rand_id
                uncertainty = np.linalg.det(Sigma_new)
                V[q_new_id].cost = V[q_rand_id].cost + uncertainty
                if uncertainty <= thresh:
                    nodes_good.append(q_new_id)
    # in the nodes_good select the one with minimal cost and return the path
    nodes_good_cost = [V[idx].cost for idx in nodes_good]
    solution = traceback(np.argmin(nodes_good_cost), V)

def obs_model(j, i):
    '''
    the observation model is 
    y_ji = l_ji(t) + v(t) 
    l_ji(t) range sensor defined as distance between robot j and target i
    v(t) Gaussian noise, sigma =  0.25 * l_ji(t) if l_ji(t) <= range else inf 
    '''
    l_ji = distance_robot_target(j, i)
    sigma = np.where(l_ji <= 2, 0.25 * l_ji, 10000)
    noise = np.random.randn(l_ji.shape)
    return l_ji + sigma * noise



# the scenario is to let multirobots to explore targets
#  hidden state is all targets positions


# Let's say the map is 15 by 15
# prior target = 7,7, with sigma = 5

# globally we have robots and targets
# are those ground truth? two versions 
targets = np.array([[6,6], [8,9], [5,2], [3,7]]) # M = 4s
robots = np.array([[1,1], [3,3]]) # N = 2
M = targets.shape[0]
N = robots.shape[0]
assignment = np.zeros((N, M),dtype=bool)
assignment[:, :N] = np.eye(N)
# x(hidden state) is the estimation of targets
x = np.zeros_like(targets) + 7 # center of the map
# p is the position of robots

A = np.eye(2)
V = np.eye(1) # measurement cov
W = np.eye(2) * 0.8 # hidden state cov
main_loop(10, hidden_Sigma=5, p0=robots)

