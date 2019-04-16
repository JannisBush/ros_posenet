#!/usr/bin/env python
import roslib
# roslib.load_manifest('my_package')
import sys	
import rospy
from std_msgs.msg import String, Int32MultiArray, Bool
from sensor_msgs.msg import Image, JointState
import almath
import numpy as np
from ros_posenet.msg import Poses, Pose, Keypoint
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed

from robot_vision.msg import PositionCommand

class publish_direction:

	def __init__(self):
    		self.img = None
		self.min_part_conf = 0.1
		self.joint_state = None
		self.image_sub = rospy.Subscriber("/camera/raw", Image, self.callback_img)
		self.keypoint_sub = rospy.Subscriber("/poses", Poses, self.callback_poses)
		self.waving_location_sub = rospy.Subscriber("waving_location", String, self.callback_waving)
		self.waving_location_pub = rospy.Publisher("waving_location", String, queue_size=10)
		self.joint_angles_pub = rospy.Publisher("joint_angles",JointAnglesWithSpeed, queue_size=10)
		self.joint_states_sub = rospy.Subscriber("joint_states",JointState, self.callback_joint_state)
		self.search_guest_sub = rospy.Subscriber("search_guest", Bool, self.callback_search)
		self.go_to_pub = rospy.Publisher("position_command", PositionCommand, queue_size=10)
		self.counter = 0
		self.counter2 = 0
		self.search= True

	def callback_search(self, search):
		self.search = search

	def callback_img(self, data):
		self.img = data

	def callback_joint_state(self,data):
		self.position = data.position[0]

	def callback_poses(self,pose_msg):
		'''
		this is the basic wavin detector. It only detects whether waving has been detected somewhere in Peppers view and sends the resullts to another function
		'''
		if self.search:
			keypoint_dict = {}
			for i in range(len(pose_msg.poses)):
			
				for keypoint in pose_msg.poses[i].keypoints:
					 if keypoint.score > self.min_part_conf:
					    keypoint_dict[keypoint.part] = (int(keypoint.position.x), int(keypoint.position.y))


			if all(k in keypoint_dict for k in ('rightWrist', 'rightShoulder', 'leftWrist', 'leftShoulder')):
				if keypoint_dict['rightWrist'][1] < keypoint_dict['rightShoulder'][1] and keypoint_dict['leftWrist'][1] < keypoint_dict['rightShoulder'][1]:
					self.waving_location_pub.publish(str(keypoint_dict["leftShoulder"][0]-int(self.img.width/2))+" "+
											str(keypoint_dict["rightShoulder"][0]-int(self.img.width/2))) 

				else: 
				
					self.counter2 = self.counter2 +1

			else:
	

				self.counter2 = self.counter2 +1
			if self.counter2 > 10:
					self.counter2 = 0
					

					#this is supposed to tell you that no waving person has been found and that you might want to turn Pepper into some other direction to find something
					#the numbers at the end of the sent String are supposed to tell you the current angle of Peppers head, not the direction he's supposed to turn to
					#it uses a type of costum message that hasn't been defined yet
					msg = PositionCommand()
					msg.command= "go"  #save?
					msg.location= "turning " + str(self.position)
					self.go_to_pub.publish(msg)
			
		
		else:
			pass
		

	def callback_waving(self, location):
		'''
		this function takes the information that waving has been detected and decides whether Pepper has to turn his head
		(when the waving is not in the center of view)
		or if he publishes the currect head
		'''
		a,b=location.data.split(" ")
		self.counter2 = 0
		locationlist=[int(a),int(b)]
		if locationlist[0] < 0 and locationlist[1] < 0:
			newpos= abs(locationlist[0]-locationlist[1])
			if newpos > 10:
				newpos = 10
                        msg = JointAnglesWithSpeed()
                        msg.joint_names=["HeadYaw"]
			msg.joint_angles=[newpos*almath.TO_RAD]
			msg.speed=0.05
			msg.relative=1
			self.joint_angles_pub.publish(msg)




		elif locationlist[0] > 0 and locationlist[1] > 0:
			newpos= abs(locationlist[0]-locationlist[1])
			if newpos > 10:
				newpos = 10
			msg = JointAnglesWithSpeed()
                        msg.joint_names=["HeadYaw"]
			msg.joint_angles=[-newpos*almath.TO_RAD]
			msg.speed=0.05
			msg.relative=1
			self.joint_angles_pub.publish(msg)

		else:
			self.counter = self.counter +1
			
			
			#this is supposed to tell you that a waving person has been found in the direction he's looking into
			#the numbers at the end of the sent String are supposed to tell you the current angle of Peppers head, not the direction he's supposed to turn to
			#it uses a type of costum message that hasn't been defined yet
			
			msg = PositionCommand()
			msg.command= "go"
			msg.location= "waving " + str(self.position)
			self.go_to_pub.publish(msg)
			




def main(args):
  ic = publish_direction()
  rospy.init_node('publish_direction', anonymous=False)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")



if __name__ == '__main__':
    main(sys.argv)
