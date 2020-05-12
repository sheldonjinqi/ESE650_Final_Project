import numpy as np
import random
from collections import defaultdict
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from tqdm import tqdm
import collision_check
import matplotlib.patches as patches

# initialization
n_max = 85000  # 85000 works
sensing_range = 2
prior_mean = np.array([5.5, 6])
prior_mean_1 = np.array([5.5, 6])
prior_cov_1 = np.array([[0.25, 0], [0, 0.25]])
prior_mean_2 = np.array([8, 9])
prior_cov_2 = np.array([[0.25, 0], [0, 0.25]])
prior_mean_3 = np.array([2.7, 2])
prior_cov_3 = np.array([[0.25, 0], [0, 0.25]])
prior_mean_4 = np.array([3, 7])
prior_cov_4 = np.array([[0.25, 0], [0, 0.25]])

prior_mean_5 = np.array([2, 4])
prior_mean_6 = np.array([5,1])





prior_cov = np.eye(2)*1
p_init = np.array([0.0, 0.0])
q_init = [p_init, prior_cov, np.linalg.det(prior_cov), 0, 0, 0]
V = [q_init]

initial_det = np.linalg.det(prior_cov)
cost_list = [initial_det]
step_size = 0.1
control_in = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1], [0, 0]]) * step_size
obstacle_space = [[2, 8], [35, 15], [5, 65], [95, 2]]
orDict = defaultdict(list)
orDict[str(p_init[0]) + '.' + str(p_init[1])].append(q_init)
depthDict = defaultdict(list)
depthDict[(q_init[5])].append(q_init)



obstacle_1 = np.array([[1, 3], [2, 3], [2, 1], [1, 1]])
obstacle_2 = np.array([[4, 8], [5, 8], [5, 5], [4, 5]])
obstacle_3 = np.array([[6.2, 7], [7, 7], [7, 5], [6.2, 5]])
obstacle_4 = np.array([[6.2, 4], [7, 4], [7, 3], [6.2, 3]])
obstacle_5 = np.array([[5.2, 4.8], [6, 4.8], [6, 4.2], [5.2, 4.2]])
obstacle_6 = np.array([[3, 3], [4, 3], [4, 0.5], [3, 0.5]])
obstacle_7 = np.array([[6.2, 11], [7, 11], [7, 8], [6.2, 8]])
obstacle_list = [obstacle_1, obstacle_2, obstacle_3,obstacle_4,obstacle_5,obstacle_6,obstacle_7]
(fig, axes) = plt.subplots(nrows=1, ncols=1, sharex=True, num='AIA* Plot')
ax = axes
for obstacle in obstacle_list:
    dx = np.abs(obstacle[1][0] - obstacle[0][0])
    dy = np.abs(obstacle[1][1] - obstacle[2][1])
    bot_left = obstacle[-1]
    ax.add_patch(patches.Rectangle(bot_left, dx, dy, fill=False))




class node:

    def __init__(self, position, cov, cost, parent, control, distance):
        self.position = position
        self.parent = parent
        self.cost = cost
        self.cov = cov
        self.control = control
        self.distance = distance


class robot:

    def __init__(self, position, control=None, target=None,done = True, stay = False):
        self.position = position
        self.control = control
        self.target = target
        self.done = done
        self.stay = stay


def traceback(current_node):
    path_list = []
    while current_node.parent != None:
        path_list.append(current_node.position)
        current_node = current_node.parent
        # print(f'current node: {current_node.position}')
        # if (current_node.parent) != None:
            # print(f'parent node: {(current_node.parent).position}')
    path_list.append(current_node.position)
    # print(f'end point {current_node.position}')
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


node_root = node(p_init, prior_cov, np.linalg.det(prior_cov), None, None, 0)

color = iter(cm.rainbow(np.linspace(0, 1, 25)))
sampled_pt_pool = []
robot_plot = np.array([[0,0],[8,0]])
target_list = np.vstack((prior_mean_1,prior_mean_2,prior_mean_3,prior_mean_4,prior_mean_5,prior_mean_6))
ax.scatter(target_list[:, 0], target_list[:, 1], color='blue', marker='o')
ax.scatter(robot_plot[:, 0], robot_plot[:, 1], color='blue', marker='x')
robot_1 = robot(np.array([0,0]))
robot_2 = robot(np.array([8,0]))
robot_list = [robot_1,robot_2]
robot_list = [robot_1]
M = target_list.shape[0]
N = len(robot_list)
completed_task = 0
assignment = np.zeros((N, M), dtype=bool)




# for i in range(target_list.shape[0]):
max_depth = 0
S_list = []
depthDict_list = []
node_end = []
nodes_good = []
target = []

counter = np.ones(2)
#initialize both V and depthdict for each robot
for i in range(N):
    node_end.append([])
    nodes_good.append([])
    target.append([])
    robot_pos = robot_list[i].position
    robot_node = node(robot_pos,prior_cov,np.linalg.det(prior_cov),None,None,0)
    S = {}
    S[str(robot_node.position[0]) + '.' + str(robot_node.position[1])] = robot_node
    depthDict = defaultdict(list)
    depthDict[robot_node.distance].append(robot_node)
    S_list.append(S)
    depthDict_list.append(depthDict)
# for single robot only
# S = {}
# S[str(node_root.position[0]) + '.' + str(node_root.position[1])] = node_root
#
# depthDict = defaultdict(list)
# depthDict[node_root.distance].append(node_root)
#
# distance_list = np.linalg.norm(target_list - node_root.position, axis=1)
# prior_mean = target_list[np.argmin(distance_list)]

# prior_mean = target_list[np.argmin(distance_list)]
# target_list = np.delete(target_list, np.argmin(distance_list), 0)
# distance_list = np.linalg.norm(target_list, axis=1)
# print(prior_mean)
node_parent = node_root
random_pts = []
best_cost = 1000
N, M = assignment.shape
# list used to store end nodes

# main loop for the algorithm

S = {}
S[str(node_root.position[0]) + '.' + str(node_root.position[1])] = node_root
depthDict = defaultdict(list)
depthDict[node_root.distance].append(node_root)

ongoing_task = []
completed_task_list = []

for i in tqdm(range(0,  1000000)):

    # print(f'iteration# :{i}')
    #compute the robot that needs to update their target
    # robot_update_list = [[robot] for robot in robot_list]
    #loop through robots

    if len(completed_task_list) == M:
        print('all task completed ')
        break
    for j in range(N):
        # print(f"robot {j}")
        if j == 0:
            c = 'g'
        else:
            c = 'r'

        # max_depth for individual robot, should be different for each robot
        max_depth_test = 0
        for key in depthDict_list[j]:
            if key > max_depth_test:
                max_depth_test = key
        # print(max_depth_test)
        # if i % 2000 ==0:
        #     print(max_depth_test)
        # check if robot is still working on a target, done = True if robot is free
        #also checks if there is target needs to bre tracked
        # if the robot is availble, assign the closest available target to it
        if robot_list[j].done and len(target_list) > 0 :
            distance_list = np.linalg.norm(target_list - robot_list[j].position, axis=1)
            target[j] = target_list[np.argmin(distance_list)]
            target_list = np.delete(target_list, np.argmin(distance_list), 0)
            robot_list[j].done = False
            print('')
            print(f'robot {j} is assigned with new target{target[j]}')
            print(f'robot {j} start position: {robot_list[j].position}')
            print(f'initial hidden state covariance: {robot_node.cov}')
            ongoing_task.append(target[j])
        # if not ongoing_task:
        #     print('all task all completed')
        #     exit()
        # probabilty of picking between random node and the furthest node
        if  robot_list[j].stay and  ongoing_task:
            # print(f'robot {j} is assigned to help finish task {ongoing_task}')
            target[j] = ongoing_task[0]
        pdf_s = np.random.choice([0, 1], p=[0.5, 0.5])
        pdf_s = 0

        # if robot_list[j].stay:
        #     # print('break')
        #     break

        if (i-counter[j]) <300 :
            pdf_s = 1
        else:
            # print('random mode ')
            pdf_s = np.random.choice([0, 1], p=[0.8, 0.2])
        # if i > 5000:
        #     pdf_s = np.random.choice([0, 1], p=[0.8, 0.2])
        #check the max_depth used in here
        data_set = [list(S_list[j].values()), depthDict_list[j].get(max_depth_test)]
        # data_set = [list(S.values()), depthDict.get(max_depth_test)]

        set = data_set[pdf_s]
        if set == None:
            print('bug')
            print(f'max depth: {max_depth}')
            print(pdf_s)
            print(data_set)
        #srandomly sample a existing node
        node_random = random.choice(set)
        random_pts.append(node_random.position)


        geo_distance = np.linalg.norm(target[j] - (control_in + node_random.position), axis=1)

        dis2target = np.linalg.norm(target[j] - node_random.position)
        # sample a control input
        if dis2target > 0.5:
            # probability of taking a random action or the action to the target
            pdf_u = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            pdf_u = 0

        small_d_group = control_in[np.argmin(geo_distance), :]

        control_set = [control_in[np.random.randint(8), :], small_d_group]
        # pdf_u = 0

        #randomly sampling a control input
        control_rand = control_set[pdf_u]
        # location of the new sampled node
        p_new = np.round(node_random.position + control_rand, decimals=2)
        # print(prior_mean)
        # print(control_in + node_random.position)
        # print(small_d_group)
        # exit()

        if not collision_check.check_collision(node_random.position, p_new, obstacle_list):
            # plt.plot(p_new[0], p_new[1],color = c ,marker = 'x')
            # plt.pause(1e-8)
            #sanity check by plotting the sampled points
            # ax.scatter(p_new[0], p_new[1], color=c, marker='x')
            dis = np.linalg.norm(node_random.position + control_rand - target[j])
            # take range measurement if within sensing range (2m) from the predicted target position
            if dis <= sensing_range and not collision_check.check_collision(p_new, target[j], obstacle_list):
                cov_new = ekf(p_new, node_random.cov)
                cost_tmp = np.linalg.det(cov_new)
                if np.isnan(cost_tmp):
                    cost_tmp = np.nan_to_num(cost_tmp)

                cost_updated = cost_tmp + node_random.cost

                #create a new node object for the sampled node
                new_node = node(p_new, cov_new, cost_updated, node_random, control_rand, node_random.distance + 1)

                key_tmp = str(new_node.position[0]) + '.' + str(new_node.position[1])
                # check if the generated node exists
                if (key_tmp) not in S_list[j].keys():
                    # S_list[j][str(new_node.position[0]) + '.' + str(new_node.position[1])] = new_node
                    # depthDict_list[j][new_node.distance].append(new_node)
                    # trial
                    S_list[j][str(new_node.position[0]) + '.' + str(new_node.position[1])] = new_node
                    depthDict_list[j][new_node.distance].append(new_node)
                    #might not be necessary
                    if new_node.distance > max_depth:
                        max_depth = new_node.distance

                elif new_node.cost < S_list[j].get(key_tmp).cost:
                    # S_list[j].update(key_tmp=new_node)
                    # depthDict_list[j][new_node.distance].append(new_node)
                    #trial
                    S_list[j].update(key_tmp=new_node)
                    depthDict_list[j][new_node.distance].append(new_node)
                    #might not be necessary
                    if new_node.distance > max_depth:
                        max_depth = new_node.distance
                # check the terminal condition
                # if np.linalg.det(new_node.cov) < 1e-4:
                    # print('check')
                # if np.linalg.det(new_node.cov) < 1e-4:
                #     print('good 1e-4')
                #     exit()
                if np.linalg.det(new_node.cov) < 5e-4:  # 5e-5 works
                    # best_cost = 10
                    #update the robot position for starting a new loop looking for new target
                    # print('good 1e-5')
                    # exit()
                    node_end[j].append(new_node)
                    if new_node.cost < best_cost:
                        # print(new_node)
                        best_node = new_node
                        best_cost = new_node.cost


                if len(node_end[j]) == 10:
                    # print(f'Robot {j} has found 10 nodes that satisfy the terminal condition')
                    # robot_list[j].position = best_node.position
                    # robot_list[j].done = True
                    # print('entered this loop 1 ')
                    # print(f'robot {j} found assigned target at iteration: {i}')
                    counter[j] = i
                    robot_list[j].done = True
                    robot_list[j].position = best_node.position
                    # re initiate the node_end list
                    node_end[j] = []
                    # print(f'robot {j} found target at {best_node.position}')
                    robot_node = node(best_node.position, prior_cov, np.linalg.det(prior_cov), None, None, 0)
                    S_list[j] = {}
                    S_list[j][str(robot_node.position[0]) + '.' + str(robot_node.position[1])] = robot_node
                    depthDict_list[j] = defaultdict(list)
                    depthDict_list[j][robot_node.distance].append(robot_node)
                    # S_list[j] = S
                    # depthDict_list[j] = depthDict
                    # path = traceback(best_node)
                    # path_array = np.asarray(path)
                    # c = next(color)
                    # ax.scatter(best_node.position[0],best_node.position[1],c = 'red', marker = 'x')
                    # ax.plot(path_array[:, 0], path_array[:, 1])
                    if robot_list[j].stay == False:
                        path = traceback(best_node)
                        path_array = np.asarray(path)
                        c = next(color)
                        ax.scatter(best_node.position[0], best_node.position[1], c='red', marker='x')
                        ax.plot(path_array[:, 0], path_array[:, 1])

                        completed_task += 1
                        completed_task_list.append(target[j])
                        # print(ongoing_task)
                        for l in range(len(ongoing_task)):
                            if np.array_equal(ongoing_task[l], target[j]):
                                del ongoing_task[l]
                                break
                        print(f'robot {j} found assigned target at iteration: {i}')
                        print(f' the target found is at: {target[j]}')
                        print(f'robot {j} found target at {best_node.position}')
                        print(f'robot {j} stay mode {robot_list[j].stay}')
                        print(f'ongoing task {ongoing_task}')
                        print(f' is target_list empty: {not target_list.tolist() }')
                    best_cost = 1000
                    if  not target_list.tolist():
                        # print('empty_target')
                        robot_list[j].stay = True
                        counter[j] = i
                        # print(f'robot {j} stay {robot_list[j].stay}')

                    if len(target_list) == 0 and robot_list[j].stay == False:
                        counter[j] = i
                        # print(f'no target to be assigned to robot {j}')
                        # print(f'robot done mode: {robot_list[j].done} ')
                        # print(f'robot stay mode: {robot_list[j].stay}')
                        # print(f'update counter {j} to {i}')



                    # break
            else:
                # the hidden state covatiance doesn't change when outside the sensing range
                cov_new = prior_cov
                cost_tmp = initial_det
                cost_updated = cost_tmp + node_random.cost

                new_node = node(p_new, cov_new, cost_updated, node_random, control_rand, node_random.distance + 1)
                key_tmp = str(new_node.position[0]) + '.' + str(new_node.position[1])

                if (key_tmp) not in S_list[j].keys():
                    # S_list[j][str(new_node.position[0]) + '.' + str(new_node.position[1])] = new_node
                    # depthDict_list[j][new_node.distance].append(new_node)
                    #trial
                    S_list[j][str(new_node.position[0]) + '.' + str(new_node.position[1])] = new_node
                    depthDict_list[j][new_node.distance].append(new_node)
                    if new_node.distance > max_depth:
                        max_depth = new_node.distance
                elif new_node.cost < S_list[j].get(key_tmp).cost:
                    # S_list[j].update(key_tmp=new_node)
                    # depthDict_list[j][new_node.distance].append(new_node)
                    #trial
                    S_list[j].update(key_tmp=new_node)
                    depthDict_list[j][new_node.distance].append(new_node)
                    if new_node.distance > max_depth:
                        max_depth = new_node.distance



plt.show()