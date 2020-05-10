# take end points from two lines and check if they intersect. Return True if intersect
def check_line_intersect(p11, p12, p21, p22):
    # check if two lines are parralle
    if ((p22[1] - p21[1]) * (p12[0] - p11[0]) - (p22[0] - p21[0]) * (p12[1] - p11[1])) == 0:
        return False

    uA = ((p22[0] - p21[0]) * (p11[1] - p21[1]) - (p22[1] - p21[1]) * (p11[0] - p21[0])) / \
         ((p22[1] - p21[1]) * (p12[0] - p11[0]) - (p22[0] - p21[0]) * (p12[1] - p11[1]))
    uB = ((p12[0] - p11[0]) * (p11[1] - p21[1]) - (p12[1] - p11[1]) * (p11[0] - p21[0])) / \
         ((p22[1] - p21[1]) * (p12[0] - p11[0]) - (p22[0] - p21[0]) * (p12[1] - p11[1]))

    return 0 <= uA <= 1 and 0 <= uB <= 1


# check if line segment is in collision with one obstacle
def obstacle_collision(p1, p2, obstacle):
    for i in range(len(obstacle)):
        p_obstacle_1 = obstacle[i % len(obstacle)]
        p_obstacle_2 = obstacle[(i + 1) % len(obstacle)]
        if check_line_intersect(p1, p2, p_obstacle_1, p_obstacle_2):
            return True
    return False

def check_collision(p1,p2,obstacle_list):
    for obstacle in obstacle_list:
        if obstacle_collision(p1,p2,obstacle):
            return True
    return False