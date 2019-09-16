#!/usr/bin/env python
from turtlebot_ctrl.srv import TurtleBotControl
from std_msgs.msg import Bool, Float32
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from kobuki_msgs.msg import BumperEvent
from turtlebot_ctrl.srv import TurtleBotControl, TurtleBotControlResponse
from turtlebot_ctrl.msg import TurtleBotScan
from copy import deepcopy
from tf.transformations import euler_from_quaternion, quaternion_from_euler








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

		while key != 's':


			modelstate = self.get_model_state(model_name="mobile_base")
			currx = modelstate.pose.position.x
			curry = modelstate.pose.position.y
			roll = modelstate.pose.orientation.x
			pitch = modelstate.pose.orientation.y
			yaw = modelstate.pose.orientation.z
			w = modelstate.pose.orientation.w
			orientation_list = [roll, pitch, yaw,w]
			(roll, pitch, yaw) = euler_from_quaternion(orientation_list)
			print "current positions are: "+str((currx,curry,yaw))
			

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
			#print(output)
			f_r.write(str(output)+"\n")



			
		f_r.close()
		rospy.spin()










if __name__ == "__main__":
	try:
		turtlebot_control_client = TurtlebotControlClient()
		
		turtlebot_control_client.run()
	except rospy.ROSInterruptException:
		pass



