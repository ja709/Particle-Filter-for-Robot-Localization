from __future__ import division
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import random
import math
import copy
import numpy as np

import matplotlib.patches as mpatches


#landmarks = [[-5.0, -1], [-4.8, -1], [-4.6, -1], [-4.3999999999999995, -1], [-4.199999999999999, -1], [-3.999999999999999, -1], [-3.799999999999999, -1], [-3.5999999999999988, -1], [-3.3999999999999986, -1], [-3.1999999999999984, -1], [-2.9999999999999982, -1], [-2.799999999999998, -1], [-2.599999999999998, -1], [-2.3999999999999977, -1], [-2.1999999999999975, -1], [-1.9999999999999973, -1], [-1.7999999999999972, -1], [-1.599999999999997, -1], [-1.3999999999999968, -1], [-1.1999999999999966, -1], [-5.0, 2], [-4.8, 2], [-4.6, 2], [-4.3999999999999995, 2], [-4.199999999999999, 2], [-3.999999999999999, 2], [-3.799999999999999, 2], [-3.5999999999999988, 2], [-3.3999999999999986, 2], [-3.1999999999999984, 2], [-5, -1.0], [-5, -0.8], [-5, -0.6000000000000001], [-5, -0.40000000000000013], [-5, -0.20000000000000018], [-5, -2.220446049250313e-16], [-5, 0.19999999999999973], [-5, 0.3999999999999997], [-5, 0.5999999999999996], [-5, 0.7999999999999996], [-5, 0.9999999999999996], [-5, 1.1999999999999993], [-5, 1.3999999999999995], [-5, 1.5999999999999996], [-5, 1.7999999999999994], [-3.0, 2], [-2.8, 1.6666666666666667], [-2.5999999999999996, 1.3333333333333335], [-2.3999999999999995, 1.0000000000000002], [-2.1999999999999993, 0.666666666666667], [-1.9999999999999991, 0.33333333333333365], [-1.799999999999999, 3.3306690738754696e-16], [-1.5999999999999988, -0.333333333333333], [-1.3999999999999986, -0.6666666666666663], [-1.1999999999999984, -0.9999999999999996], [2, -4.0], [2, -3.8], [2, -3.5999999999999996], [2, -3.3999999999999995], [2, -3.1999999999999993], [2, -2.999999999999999], [2, -2.799999999999999], [2, -2.5999999999999988], [2, -2.3999999999999986], [2, -2.1999999999999984], [2, -1.9999999999999982], [2, -1.799999999999998], [2, -1.5999999999999979], [2, -1.3999999999999977], [2, -1.1999999999999975], [2, -0.9999999999999973], [2, -0.7999999999999972], [2, -0.599999999999997], [2, -0.3999999999999968], [2, -0.19999999999999662], [2, 3.552713678800501e-15], [2, 0.20000000000000373], [2, 0.4000000000000039], [2, 0.6000000000000041], [2, 0.8000000000000043], [2, 1.0000000000000044], [2, 1.2000000000000046], [2, 1.4000000000000048], [2, 1.600000000000005], [2, 1.8000000000000052], [-2.0, 2], [-2.0, 1.7], [-1.8, 2], [-1.8, 1.4], [-1.6, 2], [-1.6, 1.0999999999999999], [-1.4000000000000001, 2], [-1.4000000000000001, 0.7999999999999998], [-1.2000000000000002, 2], [-1.2000000000000002, 0.49999999999999983], [-1.0000000000000002, 2], [-1.0000000000000002, 0.19999999999999984], [-0.8000000000000003, 2], [-0.8000000000000003, -0.10000000000000014], [-0.6000000000000003, 2], [-0.6000000000000003, -0.40000000000000013], [-0.40000000000000036, 2], [-0.40000000000000036, -0.7000000000000002], [-0.2000000000000004, 2], [-0.2000000000000004, -1.0000000000000002], [-4.440892098500626e-16, 2], [-4.440892098500626e-16, -1.3000000000000003], [0.1999999999999993, 2], [0.1999999999999993, -1.6000000000000003], [0.39999999999999947, 2], [0.39999999999999947, -1.9000000000000004], [0.5999999999999996, 2], [0.5999999999999996, -2.2], [0.7999999999999994, 2], [0.7999999999999994, -2.5], [0.9999999999999991, 2], [0.9999999999999991, -2.8], [1.1999999999999993, 2], [1.1999999999999993, -3.0999999999999996], [1.3999999999999995, 2], [1.3999999999999995, -3.3999999999999995], [1.5999999999999992, 2], [1.5999999999999992, -3.6999999999999993], [1.799999999999999, 2], [1.799999999999999, -3.999999999999999], [-5, -5.0], [-5, -4.8], [-5, -4.6], [-5, -4.3999999999999995], [-5, -4.199999999999999], [-5, -3.999999999999999], [-5, -3.799999999999999], [-5, -3.5999999999999988], [-5, -3.3999999999999986], [-5, -3.1999999999999984], [-5, -2.9999999999999982], [-5, -2.799999999999998], [-5, -2.599999999999998], [-5.0, -2.5], [-4.8, -2.5], [-4.6, -2.5], [-4.3999999999999995, -2.5], [-4.199999999999999, -2.5], [-3.999999999999999, -2.5], [-3.799999999999999, -2.5], [-3.5999999999999988, -2.5], [-3.3999999999999986, -2.5], [-3.1999999999999984, -2.5], [-2.9999999999999982, -2.5], [-2.799999999999998, -2.5], [-2.599999999999998, -2.5], [-2.3999999999999977, -2.5], [-2.1999999999999975, -2.5], [-1.9999999999999973, -2.5], [-1.7999999999999972, -2.5], [-1.599999999999997, -2.5], [-1.3999999999999968, -2.5], [-1.1999999999999966, -2.5], [-5.0, -5], [-4.8, -5], [-4.6, -5], [-4.3999999999999995, -5], [-4.199999999999999, -5], [-3.999999999999999, -5], [-3.799999999999999, -5], [-3.5999999999999988, -5], [-3.3999999999999986, -5], [-3.1999999999999984, -5], [-2.9999999999999982, -5], [-2.799999999999998, -5], [-2.599999999999998, -5], [-2.3999999999999977, -5], [-2.1999999999999975, -5], [-1.9999999999999973, -5], [-1.7999999999999972, -5], [-1.599999999999997, -5], [-1.3999999999999968, -5], [-1.1999999999999966, -5], [-0.9999999999999964, -5], [-0.7999999999999963, -5], [-0.5999999999999961, -5], [-0.3999999999999959, -5], [-0.19999999999999574, -5], [4.440892098500626e-15, -5], [0.20000000000000462, -5], [0.4000000000000048, -5], [0.600000000000005, -5], [0.8000000000000052, -5], [1.0000000000000053, -5], [-1.0, -2.5], [-0.8, -2.75], [-0.6000000000000001, -3.0], [-0.40000000000000013, -3.25], [-0.20000000000000018, -3.5], [-2.220446049250313e-16, -3.75], [0.19999999999999973, -4.0], [0.3999999999999997, -4.25], [0.5999999999999996, -4.5], [0.7999999999999996, -4.75], [0.9999999999999996, -5.0]]
landmarks = [[20.0, 20.0], [20.0, 80.0], [20.0, 50.0],
             [50.0, 20.0], [50.0, 80.0], [80.0, 80.0],
             [80.0, 20.0], [80.0, 50.0]]

# size of one dimension (in meters)
world_size = 100.0
class particles:
    def __init__(self):
        self.x = random.random()*world_size   # robot's x coordinate
        self.y = random.random()*world_size   # robot's y coordinate
        self.orientation = random.random() * 2.0 * math.pi  # robot's orientation, may not work with gazebo's orientation
        self.forward_noise = 0
        self.turn_noise = 0
        self.sense_noise = 0

    def set(self, new_x, new_y, new_orientation):
        if new_x < 0 or new_x >= world_size:
            raise ValueError('X coordinate out of bound')
        if new_y < 0 or new_y >= world_size:
            raise ValueError('Y coordinate out of bound')
        if new_orientation < 0 or new_orientation >= 2 * math.pi:
            raise ValueError('Orientation must be in [0..2pi]')

        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def set_noise(self, new_forward_noise, new_turn_noise, new_sense_noise):
        self.forward_noise = float(new_forward_noise)
        self.turn_noise = float(new_turn_noise)
        self.sense_noise = float(new_sense_noise)


    def measurement_prob(self, measurement):
        prob = 1.0

        for i in range(len(measurement)):
            dist = math.sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            prob *= self.gaussian(dist, self.sense_noise, measurement[i])
        return prob

    def sense(self):
        z = []

        for i in range(len(landmarks)):
            dist = math.sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            dist += random.gauss(0.0, self.sense_noise)
            z.append(dist)

        return z

    def move(self, turn, forward):
        if forward < 0:
            raise ValueError('Robot cannot move backwards')
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * math.pi

        # move, and add randomness to the motion command
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + (math.cos(orientation) * dist)
        y = self.y + (math.sin(orientation) * dist)

        # cyclic truncate
        x %= world_size
        y %= world_size

        # set particle
        res = particles()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)

        return res
    def gaussian(self, mu, sigma, x):
        return math.exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / math.sqrt(2.0 * math.pi * (sigma ** 2))

def evaluation(r, p):
    sum = 0.0

    for i in range(len(p)):
        # the second part is because of world's cyclicity
        dx = (p[i].x - r.x + (world_size / 2.0)) % \
             world_size - (world_size / 2.0)
        dy = (p[i].y - r.y + (world_size / 2.0)) % \
             world_size - (world_size / 2.0)
        err = math.sqrt(dx ** 2 + dy ** 2)
        sum += err

    return sum / float(len(p))

def visualization(robot, step, p, pr, weights):
    plt.figure("Robot in the world", figsize = (15., 15.))
    plt.title('Particle filter, step ' + str(step))

    # draw coordinate grid for plotting
    grid = [0, world_size, 0, world_size]
    plt.axis(grid)
    plt.grid(b=True, which='major', color='0.75', linestyle='--')
    plt.xticks([i for i in range(0, int(world_size), 5)])
    plt.yticks([i for i in range(0, int(world_size), 5)])

    # draw particles
    for ind in range(len(p)):
        # particle
        circle = plt.Circle((p[ind].x, p[ind].y), 1., facecolor='#ffb266', edgecolor='#994c00', alpha=0.5)
        plt.gca().add_patch(circle)

        # particle's orientation
        arrow = plt.Arrow(p[ind].x, p[ind].y, 2 * math.cos(p[ind].orientation), 2 * math.sin(p[ind].orientation), alpha=1.,
                          facecolor='#994c00', edgecolor='#994c00')
        plt.gca().add_patch(arrow)

    # draw resampled particles
    for ind in range(len(pr)):
        # particle
        circle = plt.Circle((pr[ind].x, pr[ind].y), 1., facecolor='#66ff66', edgecolor='#009900', alpha=0.5)
        plt.gca().add_patch(circle)

        # particle's orientation
        arrow = plt.Arrow(pr[ind].x, pr[ind].y, 2 * math.cos(pr[ind].orientation), 2 * math.sin(pr[ind].orientation), alpha=1.,
                          facecolor='#006600', edgecolor='#006600')
        plt.gca().add_patch(arrow)

    # fixed landmarks of known locations
    for lm in landmarks:
        circle = plt.Circle((lm[0], lm[1]), 1., facecolor='#cc0000', edgecolor='#330000')
        plt.gca().add_patch(circle)

    # robot's location
    circle = plt.Circle((robot.x, robot.y), 1., facecolor='#6666ff', edgecolor='#0000cc')
    plt.gca().add_patch(circle)

    # robot's orientation
    arrow = plt.Arrow(robot.x, robot.y, 2 * math.cos(robot.orientation), 2 * math.sin(robot.orientation), alpha=0.5,
                      facecolor='#000000', edgecolor='#000000')
    plt.gca().add_patch(arrow)

    plt.savefig("figure_"+ str(step) + ".png" )
    #plt.show()
    plt.close()

def main():
    nan = False
    myrobot = particles()
    myrobot = myrobot.move(0.1, 5.0)

    n = 1000  # number of particles
    p = []  # list of particles
    for i in range(n):
        x = particles()
        x.set_noise(0.05, 0.05, 5.0)
        p.append(x)

    steps = 50  # particle filter steps

    for t in range(steps):

        # move the robot and sense the environment after that
        myrobot = myrobot.move(0.1, 5.)
        z = myrobot.sense()

        # now we simulate a robot motion for each of
        # these particles
        p2 = []

        for i in range(n):
            p2.append(p[i].move(0.1, 5.))

        p = p2

        w = []

        for i in range(n):
            w.append(p[i].measurement_prob(z))
        # resampling with a sample probability proportional
        # to the importance weight
        p3 = []

        index = int(random.random() * n)
        beta = 0.0
        mw = max(w)

        for i in range(n):
            beta += random.random() * 2.0 * mw

            while beta > w[index]:
                beta -= w[index]
                index = (index + 1) % n

            p3.append(p[index])

        # here we get a set of co-located particles
        p = p3

        print 'Step = ', t, ', Evaluation = ', evaluation(myrobot, p)

        maxindex = w.index(max(w))
        print (myrobot.x, myrobot.y)
        print (p[maxindex].x, p[maxindex].y)
        #visualization(myrobot, t, p, p3, w)
    # plt.plot([4,4,-7,-7,4], [4,-7,-7,4,4])  #the map boundary
    #
    # plt.plot([-1,-5,-5,-3,-1], [-1,-1,2,2,-1])
    # plt.plot([-2,2,2,-2], [2,2,-4,2])
    # plt.plot([-1,1.2,1.2,-5,-5,-1], [-2.5,-4,-5,-5,-2.5,-2.5])
    # plt.plot(myrobot.x, myrobot.y, "xr")
    # plt.show()
if __name__ == '__main__':
    main()