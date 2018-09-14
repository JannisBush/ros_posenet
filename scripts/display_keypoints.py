#!/usr/bin/env python
from __future__ import print_function

import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from ros_posenet.msg import Poses, Pose, Keypoint

class image_converter:

  def __init__(self):

    self.img = None
    self.min_part_conf = 0.1

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/miro/rob01/platform/caml", Image, self.callback_img)

    self.keypoint_sub = rospy.Subscriber("/poses", Poses, self.callback_poses)

  def callback_img(self, data):
    self.img = data

  def callback_poses(self, pose_msg):
    # print(poses)
    if self.img is not None:
      try:
        cv_image = self.bridge.imgmsg_to_cv2(self.img, "bgr8")
      except CvBridgeError as e:
        print(e)


      if len(pose_msg.poses) > 0:
        keypoint_dict = {}
        for keypoint in pose_msg.poses[0].keypoints:
          if keypoint.score > self.min_part_conf:
            keypoint_dict[keypoint.part] = (int(keypoint.position.x), int(keypoint.position.y))
            # draw point 
            cv2.circle(cv_image, (int(keypoint.position.x), int(keypoint.position.y)), 5, (255,0,0), -1)
          #print(keypoint)
          # cv2.circle(cv_image, (int(keypoint.position.x),int(keypoint.position.y)), 5, (0,0,255), 1)
          
          # font = cv2.FONT_HERSHEY_SIMPLEX
          # cv2.putText(cv_image, keypoint.part, (int(keypoint.position.x),int(keypoint.position.y)), font, 1, (0,0,255), 1, cv2.LINE_AA)

        # information for skeleton
        connected_part_names = [
        ['leftHip', 'leftShoulder'], ['leftElbow', 'leftShoulder'],
        ['leftElbow', 'leftWrist'], ['leftHip', 'leftKnee'],
        ['leftKnee', 'leftAnkle'], ['rightHip', 'rightShoulder'],
        ['rightElbow', 'rightShoulder'], ['rightElbow', 'rightWrist'],
        ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle'],
        ['leftShoulder', 'rightShoulder'], ['leftHip', 'rightHip']]

        # draw skeleton
        for connected_part1, connected_part2 in connected_part_names:
          if all(k in keypoint_dict for k in (connected_part1, connected_part2)):
            cv2.line(cv_image, keypoint_dict[connected_part1], keypoint_dict[connected_part2], (0,0,255), 3)
            
        # detect hands up
        if all(k in keypoint_dict for k in ('rightWrist', 'rightShoulder', 'leftWrist', 'leftShoulder')):
          # print(keypoint_dict['rightWrist'],keypoint_dict['rightShoulder'])
          if keypoint_dict['rightWrist'][1] < keypoint_dict['rightShoulder'][1] and keypoint_dict['leftWrist'][1] < keypoint_dict['rightShoulder'][1]:
            # hands up detected
            # print('HandsUP')
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cv_image, 'HandsUP', (0,20), font, 1, (0,0,255), 1, cv2.LINE_AA)

        # detect high knees right
        if all(k in keypoint_dict for k in ('rightHip', 'rightKnee')):
          if keypoint_dict['rightKnee'][1] < keypoint_dict['rightHip'][1] :
            # high knee right detected
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cv_image, 'HighKneeRight', (0,150), font, 1, (0,0,255), 1, cv2.LINE_AA)
        # detect high knees left
        if all(k in keypoint_dict for k in ('leftHip', 'leftKnee')):
          if keypoint_dict['leftKnee'][1] < keypoint_dict['leftHip'][1] :
            # high knee left detected
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cv_image, 'HighKneeLeft', (150,150), font, 1, (0,0,255), 1, cv2.LINE_AA)

      cv2.imshow("Image window", cv_image)
      cv2.waitKey(3)


def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)