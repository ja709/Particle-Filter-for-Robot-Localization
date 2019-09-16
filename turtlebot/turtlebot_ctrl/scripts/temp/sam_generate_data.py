#!/usr/bin/env python
from __future__ import division
from turtlebot_ctrl.srv import TurtleBotControl
from std_msgs.msg import Bool, Float32
import rospy
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import random
import math
import time
import copy
import matplotlib.patches as mpatches
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from kobuki_msgs.msg import BumperEvent
from turtlebot_ctrl.srv import TurtleBotControl, TurtleBotControlResponse
from turtlebot_ctrl.msg import TurtleBotScan
from copy import deepcopy
from tf.transformations import euler_from_quaternion, quaternion_from_euler


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






class TurtlebotControlClient:
	def __init__(self):
		rospy.init_node("turtlebot_control_client")

		rospy.wait_for_service("turtlebot_control")
		self.turtlebot_control_service = rospy.ServiceProxy("turtlebot_control",TurtleBotControl)
		rospy.wait_for_service("/gazebo/get_model_state")
		self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state",GetModelState)

	def run(self):
		
		
		f_r = open("trajectories.txt", "w+")
		key = ""
		heading = Float32()
		heading.data = 0.0
		distance = Float32()
		return_ground_truth = Bool()
		return_ground_truth.data = True


		modelstate = self.get_model_state(model_name="mobile_base")
		currx = modelstate.pose.position.x
		curry = modelstate.pose.position.y
		roll = modelstate.pose.orientation.x
		pitch = modelstate.pose.orientation.y
		yaw = modelstate.pose.orientation.z
		w = modelstate.pose.orientation.w
		orientation_list = [roll, pitch, yaw,w]
		(roll, pitch, yaw) = euler_from_quaternion(orientation_list)

		heading.data = -np.pi/2
		time.sleep(1)
		distance.data = 1.0
		print("Heading: "+str(heading))
		print("Distance: "+str(distance))
		f_r.write("Heading: "+str(heading)+"\n")
		f_r.write("Distance: "+str(distance)+"\n")
		output = self.turtlebot_control_service(heading, distance, return_ground_truth)
		#print(output)
		f_r.write(str(output)+"\n")		
		time.sleep(1)
		distance.data = 1.0
		print("Heading: "+str(heading))
		print("Distance: "+str(distance))
		f_r.write("Heading: "+str(heading)+"\n")
		f_r.write("Distance: "+str(distance)+"\n")
		output = self.turtlebot_control_service(heading, distance, return_ground_truth)
		#print(output)
		f_r.write(str(output)+"\n")
		time.sleep(1)
		heading.data = -np.pi/4
		time.sleep(1)
		distance.data = 1.414
		print("Heading: "+str(heading))
		print("Distance: "+str(distance))
		f_r.write("Heading: "+str(heading)+"\n")
		f_r.write("Distance: "+str(distance)+"\n")
		output = self.turtlebot_control_service(heading, distance, return_ground_truth)
		#print(output)
		f_r.write(str(output)+"\n")
		time.sleep(1)
		heading.data = 0
		time.sleep(1)
		distance.data = 1
		print("Heading: "+str(heading))
		print("Distance: "+str(distance))
		f_r.write("Heading: "+str(heading)+"\n")
		f_r.write("Distance: "+str(distance)+"\n")
		output = self.turtlebot_control_service(heading, distance, return_ground_truth)
		#print(output)
		f_r.write(str(output)+"\n")
		time.sleep(1)
		distance.data = 1
		print("Heading: "+str(heading))
		print("Distance: "+str(distance))
		f_r.write("Heading: "+str(heading)+"\n")
		f_r.write("Distance: "+str(distance)+"\n")
		output = self.turtlebot_control_service(heading, distance, return_ground_truth)
		#print(output)
		f_r.write(str(output)+"\n")
		time.sleep(1)
		distance.data = 1
		print("Heading: "+str(heading))
		print("Distance: "+str(distance))
		f_r.write("Heading: "+str(heading)+"\n")
		f_r.write("Distance: "+str(distance)+"\n")
		output = self.turtlebot_control_service(heading, distance, return_ground_truth)
		#print(output)
		f_r.write(str(output)+"\n")
		time.sleep(1)
		distance.data = 1
		print("Heading: "+str(heading))
		print("Distance: "+str(distance))
		f_r.write("Heading: "+str(heading)+"\n")
		f_r.write("Distance: "+str(distance)+"\n")
		output = self.turtlebot_control_service(heading, distance, return_ground_truth)
		#print(output)
		f_r.write(str(output)+"\n")
		time.sleep(1)
		distance.data = 1
		print("Heading: "+str(heading))
		print("Distance: "+str(distance))
		f_r.write("Heading: "+str(heading)+"\n")
		f_r.write("Distance: "+str(distance)+"\n")
		output = self.turtlebot_control_service(heading, distance, return_ground_truth)
		#print(output)
		f_r.write(str(output)+"\n")
		time.sleep(1)
		heading.data = -np.pi/4
		time.sleep(1)
		distance.data = 1.414
		print("Heading: "+str(heading))
		print("Distance: "+str(distance))
		f_r.write("Heading: "+str(heading)+"\n")
		f_r.write("Distance: "+str(distance)+"\n")
		output = self.turtlebot_control_service(heading, distance, return_ground_truth)
		#print(output)
		f_r.write(str(output)+"\n")

			
		f_r.close()
		rospy.spin()










if __name__ == "__main__":
	main()	
	try:
		turtlebot_control_client = TurtlebotControlClient()
		turtlebot_control_client.run()
	except rospy.ROSInterruptException:
		pass



