# ROS PoseNet
ROS wrapper for PoseNET library in NodeJS
PoseNet demo: [Here](https://storage.googleapis.com/tfjs-models/demos/posenet/camera.html)

* Current implementation is for experiments only *

### Requirements
  * ROS Kinetic
  * NodeJS 8.x 
  * Cuda 9.0 + cuDNN 7.1 (for GPU acceleration only)
  
### Installation
  *  Clone to ROS workspace and build
  * `npm install` inside the package folder
  
### Configuring

Following ROS parameters should be set:
  * `gpu: (true / false)` - Specifies if GPU acceleration should be used
  * `topic` - Uncompressed RGB8 encoded image topic
  * `out_topic` -  specifies topic for result output. Output topic publishes JSON string that needs to be decoded as `std_msgs/String` message
  *  Algorithm parameters to adjust performance. See [launch file](launch/posenet.launch) for full list> References could be found [PoseNet Official Github](https://github.com/tensorflow/tfjs-models/tree/master/posenet#inputs-2)
  
### Running
  * `roslaunch ros_posenet posenet.launch` or `rosrun ros_posenet posenet.js`
  
### Limitations
 * Only multiple pose detection implemented
 * Requires internet to download the model weights
 * Only ROS tpopics with RGB8 encoding are supported as inputs
