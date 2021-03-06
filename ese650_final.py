import numpy as np
import random
from collections import defaultdict
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from tqdm import tqdm
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
q_init = [p_init, prior_cov, np.linalg.det(prior_cov), 0, 0, 0]
V = [q_init]

initial_det = np.linalg.det(prior_cov)
cost_list = [initial_det]
step_size = 0.1
control_in = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1],[0,0]]) * step_size
obstacle_space = [[2, 8], [35, 15], [5, 65], [95, 2]]
orDict = defaultdict(list)
orDict[str(p_init[0]) + '.' + str(p_init[1])].append(q_init)
depthDict = defaultdict(list)
depthDict[(q_init[5])].append(q_init)


class node:

    def __init__(self, position, cov, cost, parent, control, distance):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.cov = cov
        self.control = control
        self.distance = distance


def traceback(current_node):
    path_list = []
    while current_node.parent != None:
        path_list.append(current_node.position)
        current_node = current_node.parent
    path_list.append(current_node.position)
    return path_list


def ekf(p, sigma_prior):
    x_dis = (prior_mean[0] - p[0]) ** 2
    y_dis = (prior_mean[1] - p[1]) ** 2
    geo_dis = np.sqrt(x_dis+y_dis)
    if geo_dis == 0:
        h = np.zeros(2).reshape(1,-1)
    else:
        h = np.array([[(prior_mean[0] - p[0]) / geo_dis],
                  [(prior_mean[1] - p[1]) / geo_dis]]).T
    sigma_measurement = 0.04


    s = h @ sigma_prior @ (h.T) + sigma_measurement
    k_gain = sigma_prior @ (h.T) @ np.linalg.inv(s)

    cov_update = (np.eye(2) - k_gain @ h) @ sigma_prior

    return cov_update


node_root = node(p_init, prior_cov, np.linalg.det(prior_cov), None, None, 0)
target_list = np.vstack((prior_mean_1,prior_mean_2,prior_mean_3,prior_mean_4)) - p_init
plot_target = target_list
distance_list = np.linalg.norm(target_list, axis=1)
color=iter(cm.rainbow(np.linspace(0,1,4)))
sampled_pt_pool = [ ]
for i in range(target_list.shape[0]):
    max_depth = 0
    S = {}
    S[str(node_root.position[0]) + '.' + str(node_root.position[1])] = node_root

    depthDict = defaultdict(list)
    depthDict[node_root.distance].append(node_root)

    prior_mean = target_list[np.argmin(distance_list)]
    target_list = np.delete(target_list, np.argmin(distance_list), 0)
    distance_list = np.linalg.norm(target_list, axis=1)
    nodes_good = []

    # print(prior_mean)
    node_parent = node_root
    random_pts = []
    best_cost = 10
    for i in tqdm(range(0, 4*n_max)):
        # probabilty of picking between random node and the furthest node
        pdf_s = np.random.choice([0, 1], p=[0.5, 0.5])
        pdf_s = 0
        data_set = [list(S.values()), depthDict.get(max_depth)]
        set = data_set[pdf_s]
        # set = list(S.values())

        node_random = random.choice(set)
        random_pts.append(node_random.position)

        geo_distance = np.linalg.norm(prior_mean - (control_in + node_random.position), axis=1)
        # geo_distance = np.linalg.norm(prior_mean - (control_in + np.array([6,6])), axis=1)

        dis_old = np.linalg.norm(prior_mean - node_random.position)
        # sample a control input
        if dis_old > 2:
            # probability of taking a random action or the action to the target
            pdf_u = np.random.choice([0, 1], p=[0.5, 0.5])
        else:
            pdf_u = 0
        small_d_group = control_in[np.argmin(geo_distance), :]

        control_set = [control_in[np.random.randint(8), :], small_d_group]
        # pdf_u = 0
        control_rand = control_set[pdf_u]
        # location of the sampled node
        p_new = np.round(node_random.position + control_rand, decimals=2)
        sampled_pt_pool.append(p_new)
        # print(prior_mean)
        # print(control_in + node_random.position)
        # print(small_d_group)
        # exit()

        if p_new.tolist() not in obstacle_space:
            dis = np.linalg.norm(node_random.position + control_rand - prior_mean)

            # take range measurement if within 2m from the predicted target position
            if dis <= sensing_range:
                cov_new = ekf(p_new, node_random.cov)
                cost_tmp = np.linalg.det(cov_new)


                if np.isnan(cost_tmp):
                    cost_tmp = np.nan_to_num(cost_tmp)
                cost_updated = cost_tmp + node_random.cost
                new_node = node(p_new, cov_new, cost_updated, node_random, control_rand, node_random.distance + 1)
                key_tmp = str(new_node.position[0]) + '.' + str(new_node.position[1])
                # check if the generated configuration exists
                if (key_tmp) not in S.keys():
                    S[str(new_node.position[0]) + '.' + str(new_node.position[1])] = new_node
                    depthDict[new_node.distance].append(new_node)
                    if new_node.distance > max_depth:
                        max_depth = new_node.distance
                elif new_node.cost < S.get(key_tmp).cost:
                    S.update(key_tmp=new_node)
                    depthDict[new_node.distance].append(new_node)
                    if new_node.distance > max_depth:
                        max_depth = new_node.distance
                # check the terminal condition
                if np.linalg.det(new_node.cov) < 1e-5: #5e-5 works
                    # best_cost = 10
                    if new_node.cost < best_cost:
                        # print(new_node)
                        best_node = new_node
                        best_cost = new_node.cost
                    nodes_good.append(new_node)

                    # number of nodes satisfy the terminal condition
                    if len(nodes_good) == 10:
                        print('yes')
                        print(i)
                        break
            else:
                cov_new = prior_cov
                cost_tmp = initial_det
                cost_updated = cost_tmp + node_random.cost

                new_node = node(p_new, cov_new, cost_updated, node_random, control_rand, node_random.distance + 1)
                key_tmp = str(new_node.position[0]) + '.' + str(new_node.position[1])

                if (key_tmp) not in S.keys():
                    S[str(new_node.position[0]) + '.' + str(new_node.position[1])] = new_node
                    depthDict[new_node.distance].append(new_node)
                    if new_node.distance > max_depth:
                        max_depth = new_node.distance
                elif new_node.cost < S.get(key_tmp).cost:
                    S.update(key_tmp=new_node)
                    depthDict[new_node.distance].append(new_node)
                    if new_node.distance > max_depth:
                        max_depth = new_node.distance

            orDict[key_tmp].append(new_node)

        if len(nodes_good) == 20:
            break
    # check if a path was found
    if len(nodes_good) == 0:
        print('no path found')
        # for sampled_pt in sampled_pt_pool:
        #     plt.plot(sampled_pt[0], sampled_pt[1], marker='.')
        # plt.show()
        exit()
    else:
        q_end = best_node


    path = traceback((q_end))
    a = np.asarray(path)
    c = next(color)
    plt.plot(a[:, 0], a[:, 1],c=c)
    plt.scatter((q_end.position)[0],(q_end.position)[1],marker='.')
    node_root.position = best_node.position
    # node_root.parent = None
    # print(node_root)
    # print(best_node)
    node_root.distance = 0
    # print(q_end.cov)
    print(q_end.position)

    # path = traceback((q_end))
    # a = np.asarray(path)
    # c = next(color)
    # plt.plot(a[:, 0], a[:, 1],c=c)
    # plt.scatter((q_end.position)[0],(q_end.position)[1])


obstacle_space = [[2, 8], [5, 1], [5, 6], [9, 2]]

obstacle = np.asarray(obstacle_space)
# plt.plot(a[:, 0], a[:, 1], color='red')
plt.scatter(plot_target[:, 0], plot_target[:, 1], color='blue', marker='o')
plt.scatter(obstacle[:, 0], obstacle[:, 1], color='green', marker='x')

plt.show()
