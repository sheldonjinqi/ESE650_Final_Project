import numpy as np
import random
from collections import defaultdict
import math
from matplotlib import pyplot as plt

# initialization
n_max = 85000 #85000 works
sensing_range = 2
prior_mean = np.array([6, 6])
prior_mean_1 = np.array([6,6])
prior_cov_1 = np.array([[0.25, 0], [0, 0.25]])
prior_mean_2 = np.array([8, 9])
prior_cov_2 = np.array([[0.25, 0], [0, 0.25]])
prior_mean_3 = np.array([5, 2])
prior_cov_3 = np.array([[0.25, 0], [0, 0.25]])
prior_mean_4 = np.array([3, 7])
prior_cov_4 = np.array([[0.25, 0], [0, 0.25]])
prior_cov = np.array([[0.25, 0], [0, 0.25]])
p_init = np.array([0.0, 0.0])
q_init = [p_init, prior_cov, np.linalg.det(prior_cov), 0, 0, 0]
V = [q_init]


initial_det = np.linalg.det(prior_cov)
cost_list = [initial_det]

control_in = np.array([[0.2, 0], [-0.2, 0], [0, 0.2], [0, -0.2], [0.2, 0.2], [0.2, -0.2], [-0.2, 0.2], [-0.2, -0.2]])
obstacle_space = [[2, 8], [35, 15], [5, 65], [95, 2]]
orDict = defaultdict(list)
orDict[str(p_init[0]) + '.' + str(p_init[1])].append(q_init)
depthDict = defaultdict(list)
depthDict[(q_init[5])].append(q_init)

class node:

    def __init__(self, position, cov, cost, parent, control,distance):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.cov = cov
        self.control = control
        self.distance = distance

def traceback(current_node):
    path_list = []
    while current_node.parent != None :
        path_list.append(current_node.position)
        current_node = current_node.parent
    return path_list
def ekf(p, sigma):
    h = np.array([[(prior_mean[0] - p[0]) / np.sqrt((prior_mean[0] - p[0]) ** 2 + (prior_mean[1] - p[1]) ** 2)],
                  [(prior_mean[1] - p[1]) / np.sqrt((prior_mean[0] - p[0]) ** 2 + (prior_mean[1] - p[1]) ** 2)]]).T
    r = 0.04
    s = h @ sigma @ (h.T) + r
    k = sigma @ (h.T) @ np.linalg.inv(s)
    sigma_update = (np.eye(2) - k @ h) @ sigma
    sigma_predict = sigma_update

    return sigma_predict

node_root = node(p_init,prior_cov,np.linalg.det(prior_cov),None,None,0)
target_list = np.vstack((prior_mean_1,prior_mean_2,prior_mean_3,prior_mean_4))-p_init
plot_target = target_list
distance_list = np.linalg.norm(target_list, axis=1)

for i in range(target_list.shape[0]):
    max_depth = 0
    S = {}
    S[str(node_root.position[0])+'.'+str(node_root.position[1])] = node_root

    depthDict = defaultdict(list)
    depthDict[node_root.distance].append(node_root)

    prior_mean = target_list[np.argmin(distance_list)]
    target_list = np.delete(target_list,np.argmin(distance_list),0)
    distance_list = np.linalg.norm(target_list, axis=1)
    nodes_good=[]

    # print(prior_mean)
    node_parent = node_root
    random_pts = []
    for i in range(0, n_max):
        # probabilty of picking between random node and the furthest node
        pdf_s = np.random.choice([0, 1], p=[0.7, 0.3])
        data_set = [list(S.values()), depthDict.get(max_depth)]
        set = data_set[pdf_s]
        # set = list(S.values())

  
        node_random = random.choice(set)
        random_pts.append(node_random.position)

        geo_distance = np.linalg.norm(prior_mean-(control_in + node_random.position),axis=1)
        dis_old = np.linalg.norm(prior_mean-node_random.position)
        if dis_old > 2:
            #probability of taking a random action or the action to the target
            pdf_u = np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            pdf_u = 0
        small_d_group = control_in[np.argmin(geo_distance),:]
        control_set = [control_in[np.random.randint(8), :],small_d_group]
        control_rand = control_set[pdf_u]
        p_new = np.round(node_random.position + control_rand, decimals =2)

        if p_new.tolist() not in obstacle_space:
            dis = np.linalg.norm(node_random.position+control_rand-prior_mean)
            if dis <= sensing_range:

                # take range measurement if within 2m from the predicted target position
                det_new = ekf(p_new,node_random.cov)
                cost_tmp = np.linalg.det(det_new)
                # print(cost_tmp)
                if np.isnan(cost_tmp):
                    cost_tmp = np.nan_to_num(cost_tmp)
                # cost_updated = cost_tmp + q_rand[2]
                cost_updated = cost_tmp + node_random.cost
                new_node = node(p_new,det_new,cost_updated,node_random,control_rand,node_random.distance+1)
                key_tmp = str(new_node.position[0])+'.'+str(new_node.position[1])

                if (key_tmp) not in S.keys():
                    S[str(new_node.position[0])+'.'+str(new_node.position[1])] = new_node
                    depthDict[new_node.distance].append(new_node)
                    if new_node.distance > max_depth:
                        max_depth = new_node.distance
                elif new_node.cost < S.get(key_tmp).cost:
                    S.update(key_tmp = new_node)
                    depthDict[new_node.distance].append(new_node)
                    if new_node.distance > max_depth:
                        max_depth = new_node.distance

                if np.linalg.det(new_node.cov) < 5e-5:
                    best_cost = 10
                    if new_node.cost<best_cost:
                        # print(new_node)
                        best_node = new_node
                        best_cost = new_node.cost
                    nodes_good.append(new_node)

                    if len(nodes_good) == 10:
                        print('yes')
                        break
            else:
                cov_new = prior_cov
                cost_tmp = initial_det
                cost_updated = cost_tmp + node_random.cost

                new_node = node(p_new,cov_new,cost_updated,node_random,control_rand,node_random.distance+1)
                key_tmp = str(new_node.position[0])+'.'+str(new_node.position[1])

                if (key_tmp) not in S.keys():
                    S[str(new_node.position[0])+'.'+str(new_node.position[1])] = new_node
                    depthDict[new_node.distance].append(new_node)
                    if new_node.distance > max_depth:
                        max_depth = new_node.distance
                elif new_node.cost < S.get(key_tmp).cost:
                    S.update(key_tmp = new_node)
                    depthDict[new_node.distance].append(new_node)
                    if new_node.distance > max_depth:
                        max_depth = new_node.distance

            orDict[key_tmp].append(new_node)

        if len(nodes_good) == 10:
            break
    # check if a path was found
    if len(nodes_good) == 0:
        print('no path found')
        exit()
    else:
        q_end = best_node
    node_root = best_node
    node_root.distance = 0

    print(q_end.position)

path = traceback((q_end))
a = np.asarray(path)


obstacle_space = [[2,8],[5,1],[5,6],[9,2]]

obstacle = np.asarray(obstacle_space)
plt.plot(a[:,0],a[:,1],color = 'red')

plt.scatter(plot_target[:,0],plot_target[:,1],color='blue', marker='o')
plt.scatter(obstacle[:,0],obstacle[:,1],color='green', marker='x')
plt.show()