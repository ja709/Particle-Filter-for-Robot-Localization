#!/usr/bin/env python
from turtlebot_ctrl.srv import TurtleBotControl
from std_msgs.msg import Bool, Float32
import rospy
import numpy as np
from __future__ import division
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import random
import math
import copy
import numpy as np

n =100 #number of particles
start = [-6,1,0]
particles = {}
boundary = [(4,4),(4,-7),(-7,-7),(-7,4)]
obstacle1 = [(-1,-1),(-5,-1),(-5,2),(-3,2)]
obstacle2 = [(-2,2),(2,2),(2,-4)]
obstacle3 = [(-1,-2.5),(1.2,-4),(1.2,-5),(-5,-5),(-5,-2.5)]

nan = 11
ranges= [1.9969024658203125, 2.0711448192596436, 2.148198127746582, 2.2339370250701904, 2.3225631713867188, 2.421276807785034, 2.5317583084106445, 2.6560771465301514, 2.7968101501464844, 2.9440836906433105, 3.111342668533325, 3.302703619003296, 3.5234932899475098, 3.7807207107543945, 4.0565080642700195, 4.41288948059082, 4.8051934242248535, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 4.960227012634277, 4.509247303009033, 4.166668891906738, 3.85205078125, 3.7452032566070557, 3.7658636569976807, 3.789804220199585, 3.817359209060669, 3.844259262084961, 3.8749523162841797, 3.9046850204467773, 3.938375949859619, 3.9735679626464844, 2.25943660736084, 2.171790599822998, 2.09307861328125, 2.022090435028076]
def inside_polygon(x, y, points):

    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def outside_polygon(x,y,points):
    return not inside_polygon(x, y, points)

def distance(x1, x2, y1, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def drawparticle(x,y,orientation):
    circle = plt.Circle((x, y), 0.15, facecolor='#ffb266', edgecolor='#994c00', alpha=0.5)
    plt.gca().add_patch(circle)
    ax = plt.axes()
    ax.arrow(x, y, 0.5 * math.cos(orientation), 0.5 * math.sin(orientation), head_width=0.2, head_length=0.4, fc='#994c00', ec='#994c00')

def drawstart(x,y,orientation):
    circle = plt.Circle((x, y), 0.15, facecolor='r', edgecolor='r', alpha=0.5)
    plt.gca().add_patch(circle)
    ax = plt.axes()
    ax.arrow(x, y, 0.5 * math.cos(orientation), 0.5 * math.sin(orientation), head_width=0.2, head_length=0.4, fc='r', ec='r')

def drawprediction(x,y,orientation):
    circle = plt.Circle((x, y), 0.15, facecolor='g', edgecolor='g', alpha=0.5)
    plt.gca().add_patch(circle)
    ax = plt.axes()
    ax.arrow(x, y, 0.5 * math.cos(orientation), 0.5 * math.sin(orientation), head_width=0.2, head_length=0.4, fc='g', ec='g')

def generateParticles(n, obervedata):
    for i in range(n):
        x = random.uniform(-7, 4)
        y = random.uniform(-7,4)
        if inside_polygon(x,y, obstacle1) or inside_polygon(x,y, obstacle2) or inside_polygon(x,y, obstacle3):
            continue
        yaw = math.radians(random.randint(0, 360))
### need to find out how to obtain scan data
        particles[(x,y,yaw)] =[obervedata,0]

def randomdata():
    result = []
    for i in range(54):
        if random.randint(0,10)>4:
            result.append(nan)
        else:
            result.append(random.uniform(0.45, 10))
    return result

def calculateweight(observedata, correctdata):
    weight = 0
    for i in range(len(observedata)):
        weight = weight + math.e**(-1*abs(observedata[i]-correctdata[i]))
    return weight

def updateparticles(distance):
    start[0]=start[0]+distance
    start[1]=start[1]+distance
    for key in particles:
        particles[(key[0]+distance, key[1]+distance, key[2])] = particles.pop(key)

    for key in particles.keys():
        if inside_polygon(key[0], key[1], obstacle1) or inside_polygon(key[0], key[1], obstacle2) or inside_polygon(key[0], key[1], obstacle3) or outside_polygon(key[0], key[1], boundary):
            del particles[key]

    ##update scanning data


def iterateshow():
    # this is the axis length
    drawstart(start[0], start[1], start[2])
    plt.axis([-10, 10, -10, 10])

    # this is ploting the boundary of the map
    plt.plot([4, 4, -7, -7, 4], [4, -7, -7, 4, 4], 'b-')

    # obstacle#1
    plt.plot([-1, -5, -5, -3, -1], [-1, -1, 2, 2, -1], 'c-')
    # obstacle#2
    plt.plot([-2, 2, 2, -2], [2, 2, -4, 2], 'c-')
    # obstacle#3
    plt.plot([-1, 1.2, 1.2, -5, -5, -1], [-2.5, -4, -5, -5, -2.5, -2.5], 'c-')
    listofweight = []
    for key in particles:
        drawparticle(key[0],key[1],key[2])
        particles[key][1] = calculateweight(particles[key][0], ranges)
        listofweight.append(particles[key][1])
    max_value = max(listofweight)
    index = listofweight.index(max_value)

    for j, (key, value) in enumerate(particles.items()):
        if index == j:
            prediction = key
    # drawing the prediction
    drawprediction(prediction[0], prediction[1], prediction[2])

    plt.show()

def start():
    # this is the starting point

    drawstart(start[0],start[1],start[2])

    listofweight = []
    generateParticles(n)


    #this is the axis length
    plt.axis([-10, 10, -10, 10])

    #this is ploting the boundary of the map
    plt.plot([4, 4, -7, -7, 4], [4, -7, -7, 4, 4], 'b-')

    #obstacle#1
    plt.plot([-1,-5,-5,-3,-1], [-1,-1,2,2,-1], 'c-')
    #obstacle#2
    plt.plot([-2,2,2,-2],[2,2,-4,2],'c-')
    #obstacle#3
    plt.plot([-1,1.2,1.2,-5,-5,-1], [-2.5,-4,-5,-5,-2.5,-2.5], 'c-')

    for key in particles:
        drawparticle(key[0],key[1],key[2])
        particles[key][1] = calculateweight(particles[key][0], ranges)
        listofweight.append(particles[key][1])


    max_value = max(listofweight)
    index = listofweight.index(max_value)

    for j, (key, value) in enumerate(particles.items()):
        if index==j:
            prediction = key
    #drawing the prediction
    drawprediction(prediction[0], prediction[1], prediction[2])


    plt.show()

def convertscandata(scan_data):
	arr= str(scan_data).split(":")[1]
	arr2= arr.split("\n")[0]
	arr3= arr2.replace("[","")
	arr4= arr3.replace("]","")
	arr4=arr4.replace(" ","")
	arr5=arr4.split(",",100)
	for index,value in enumerate (arr5):
		if value=="nan":
			pass
		else:
		   	arr5[index]=float(value)
	return arr5

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
		scan_data = rospy.wait_for_message("/turtlebot_scan",TurtleBotScan)
		scan_data = convertscandata(scan_data)
		print ""
		print scan_data
		print ""
		while key != 's':
			key = raw_input("PRESS CONTROL KEYS:\n(The rotation keys rotate the turtlebot with respect to x-axis)\nl : +45 degree\na : -45 degree\nt : +90 degree\nv : -90 degree\nj : 0 degree\nf : -180 degree\nh : +135 degree\ng: -135 degree\n\nd : to move 1 cm\nm : to move sqrt(2) cm (diagonally)\n\ns : to stop\n")
			distance.data = 0.0

			if key == 'l':
				heading.data = np.pi/4
			elif key == 'a':
				heading.data = -np.pi/4
			elif key == 't':
				heading.data = np.pi/2
			elif key == 'v':
				heading.data = -np.pi/2
			elif key == 'j':
                                heading.data = 0
                        elif key == 'f':
                                heading.data = -np.pi
                        elif key == 'h':
                                heading.data = 3*np.pi/4
			elif key == 'g':
                                heading.data = -3*np.pi/4
			elif key == 'd':
				distance.data = 1.0
			elif key == 'm':
				distance.data = 1.414

			print("Heading: "+str(heading))
			print("Distance: "+str(distance))
			f_r.write("Heading: "+str(heading)+"\n")
			f_r.write("Distance: "+str(distance)+"\n")
			output = self.turtlebot_control_service(heading, distance, return_ground_truth)
			print(output)
			f_r.write(str(output)+"\n")
			
		f_r.close()
		rospy.spin()

if __name__ == "__main__":
	try:
		turtlebot_control_client = TurtlebotControlClient()
		turtlebot_control_client.run()
	except rospy.ROSInterruptException:
		pass
