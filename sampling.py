import numpy as np
import random
from collections import defaultdict
import math
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm
from tqdm import tqdm
import collision_check
import scipy
from scipy import spatial


# initialization
n_max = 85000  # 85000 works
sensing_range = 1
r = 0.15
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

step_size = 0.1
control_in = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1], [0, 0]]) * step_size
obstacle_1 = np.array([[4, 4], [5, 4], [5, 3], [4, 3]])
obstacle_2 = np.array([[4, 7], [5, 7], [5, 5], [4, 5]])

obstacle_list = [obstacle_1, obstacle_2]

obstacle_space = [[2, 8], [35, 15], [5, 65], [95, 2]]
orDict = defaultdict(list)
orDict[str(p_init[0]) + '.' + str(p_init[1])].append(q_init)
depthDict = defaultdict(list)
depthDict[(q_init[5])].append(q_init)

target_list = np.vstack((prior_mean_1, prior_mean_2, prior_mean_3, prior_mean_4))

# prior_mean_1 = np.array([2, 2])
# prior_mean_2 = np.array([2.1, 2.1])
#
# target_list = np.vstack((prior_mean_2, prior_mean_1))

plot_target = target_list
distance_list = np.linalg.norm(target_list, axis=1)
color = iter(cm.rainbow(np.linspace(0, 1, 4)))
sampled_pt_pool = []
(fig, axes) = plt.subplots(nrows=1, ncols=1, sharex=True, num='RRT* Plot')
ax = axes
ax.scatter(plot_target[:, 0], plot_target[:, 1], color='blue', marker='o')

#plot the obstacles
for obstacle in obstacle_list:
    dx = obstacle[1][0]-obstacle[0][0]
    dy = obstacle[1][1]-obstacle[2][1]
    left_bottom = obstacle[-1]
    ax.add_patch(patches.Rectangle(left_bottom,dx,dy,fill= False))



class node:

    def __init__(self, position, cov, cost, parent, control, distance,travel_distance):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.cov = cov
        self.control = control
        self.distance = distance
        self.travel_distance = travel_distance


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
    geo_dis = np.sqrt(x_dis + y_dis)
    if geo_dis == 0:
        h = np.zeros(2).reshape(1, -1)
    else:
        h = np.array([[(prior_mean[0] - p[0]) / geo_dis],
                      [(prior_mean[1] - p[1]) / geo_dis]]).T
    sigma_measurement = 0.04

    s = h @ sigma_prior @ (h.T) + sigma_measurement
    k_gain = sigma_prior @ (h.T) @ np.linalg.inv(s)

    cov_update = (np.eye(2) - k_gain @ h) @ sigma_prior

    return cov_update

def RiccatiMap(p,sigma,A,M,W,V):
    return scipy.linalg.solve_discrete_are(A.T,M.T,W,V)


def kd_near(node_list,x_new,r,nodes):
    # nodes = []
    near_nodes = []
    # for node in node_list:
    #     nodes.append(list(node.loc))
    tree = spatial.KDTree(nodes)
    near_ind = tree.query_ball_point(x_new.position, r)
    near_nodes = [node_list[i] for i in near_ind]
    # for i in near_ind:
    #     near_nodes.append(node_list[i])
    return near_nodes

node_root = node(p_init, prior_cov, np.linalg.det(prior_cov), None, None, 0,0)


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
    for i in tqdm(range(0, 5000)):
        # probabilty of picking between random node and the furthest node
        node_list = list(S.values())
        node_position_list = [list(x.position) for x in node_list]

        pdf_s = np.random.choice([0, 1], p=[0.4, 0.6])

        data_set = [list(S.values()), depthDict.get(max_depth)]
        set = data_set[pdf_s]
        node_random = random.choice(set)
        # random_pts.append(node_random.position)

        geo_distance = np.linalg.norm(prior_mean - (control_in + node_random.position), axis=1)

        # geo_distance = np.linalg.norm(prior_mean - node_random.position)

        # geo_distance = np.linalg.norm(prior_mean - (control_in + np.array([6,6])), axis=1)

        dis_old = np.linalg.norm(prior_mean - node_random.position)
        # sample a control input
        if dis_old > 1:
            # probability of taking a random action or the action to the target
            pdf_u = np.random.choice([0, 1], p=[0.4, 0.6])
        else:
            pdf_u = np.random.choice([0, 1], p=[0.7, 0.3])

        if pdf_u == 0 :
            c = 'g'
        else:
            c = 'r'

        small_d_group = control_in[np.argmin(geo_distance), :]

        control_set = [control_in[np.random.randint(8), :], small_d_group]
        control_rand = control_set[pdf_u]
        # location of the sampled node
        p_new = np.round(node_random.position + control_rand, decimals=2)


        sampled_pt_pool.append(p_new)

        if not collision_check.check_collision(node_random.position, p_new, obstacle_list):
            #     print('no collision')
            #
            # if p_new.tolist() not in obstacle_space:

            travel_distance = node_random.travel_distance +  np.linalg.norm(control_rand)
            ax.scatter(p_new[0],p_new[1],color = c, marker=',',lw = 0 , s =1 )

            dis = np.linalg.norm(node_random.position + control_rand - prior_mean)
            # take range measurement if within 2m from the predicted target position
            if dis <= sensing_range:
                # print('inside measument range')
                cov_new = ekf(p_new, node_random.cov)
                cost_tmp = np.linalg.det(cov_new)

                if np.isnan(cost_tmp):
                    cost_tmp = np.nan_to_num(cost_tmp)
                cost_updated = cost_tmp + node_random.cost
                new_node = node(p_new, cov_new, cost_updated, node_random, control_rand, node_random.distance + 1,travel_distance)
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
                if np.linalg.det(new_node.cov) < 1e-5:  # 5e-5 works
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

            # sampled location is not in the measurment range of sensor
            else:


                # print('outside measument range')
                cov_new = prior_cov
                cost_tmp = initial_det
                cost_updated = cost_tmp + node_random.cost

                new_node = node(p_new, cov_new, cost_updated, node_random, control_rand, node_random.distance + 1,travel_distance)
                key_tmp = str(new_node.position[0]) + '.' + str(new_node.position[1])

                # find the near nodes for rewiring
                near_nodes = kd_near(node_list, new_node, r , node_position_list)
                test_list =[]
                for neighbor_node in near_nodes:
                    test_list.append(neighbor_node.position)
                    if not collision_check.check_collision(neighbor_node.position,new_node.position,obstacle_list):
                        tmp_distance = neighbor_node.distance + 1
                        tmp_travel_distance = neighbor_node.travel_distance + np.linalg.norm(neighbor_node.position - new_node.position)
                        if tmp_distance < new_node.distance or tmp_travel_distance < new_node.travel_distance:
                            new_node.parent = neighbor_node
                            new_node.distance = tmp_distance
                            new_node.travel_distance = tmp_travel_distance
                # print(f"neighbor nodes: {test_list}")
                for neighbor_node in near_nodes:
                    if not collision_check.check_collision(neighbor_node.position, new_node.position, obstacle_list):
                        if neighbor_node.distance > new_node.distance + 1 or neighbor_node.travel_distance > new_node.travel_distance + np.linalg.norm(neighbor_node.position - new_node.position):
                            neighbor_node.parent = new_node
                            neighbor_node.distance = new_node.distance + 1
                            neighbor_node.travel_distance = new_node.travel_distance + np.linalg.norm(neighbor_node.position - new_node.position)





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
            print(len(nodes_good))
            break
    # check if a path was found
    if len(nodes_good) == 0:
        print('no path found')
        path = traceback((new_node))
        a = np.asarray(path)
        c = next(color)
        ax.plot(a[:, 0], a[:, 1], c=c)
        plt.show()
        # for sampled_pt in sampled_pt_pool:
        #     plt.plot(sampled_pt[0], sampled_pt[1], marker='.')
        # plt.show()
        exit()
    else:
        q_end = best_node

    path = traceback((q_end))
    a = np.asarray(path)
    c = next(color)
    ax.plot(a[:, 0], a[:, 1], c=c)
    ax.scatter((q_end.position)[0], (q_end.position)[1], marker='.')
    node_root.position = best_node.position
    # node_root.parent = None
    # print(node_root)
    # print(best_node)
    node_root.distance = 0
    # print(q_end.cov)
    print(q_end.position)

obstacle_space = [[2, 8], [5, 1], [5, 6], [9, 2]]
obstacle_1 = np.array([[4, 4], [5, 4], [5, 3], [4, 3]])

obstacle = np.asarray(obstacle_space)
# plt.plot(a[:, 0], a[:, 1], color='red')
ax.scatter(plot_target[:, 0], plot_target[:, 1], color='blue', marker='o')


plt.show()
