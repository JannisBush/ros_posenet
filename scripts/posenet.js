#!/usr/bin/env node

'use strict'

global.XMLHttpRequest = require("xhr2");
const assert = require("assert");
// Main requirements
const tf = require('@tensorflow/tfjs');
const rosnodejs = require('rosnodejs');
const stringify = require('json-stringify');
// Requires the std_msgs message and sensor_msgs packages
const sensor_msgs = rosnodejs.require('sensor_msgs').msg;
const StringMsg = rosnodejs.require('std_msgs').msg.String;
const pose_msgs = rosnodejs.require('ros_posenet').msg;
const PosesMsg = rosnodejs.require('ros_posenet').msg.Poses;

// MobileNet Image Classification
// const mobilenet = require('@tensorflow-models/mobilenet');
// const loadModel = async path => {
//   const mn = new mobilenet.MobileNet(1, 1);
//   mn.path = `file://${path}`
//   await mn.load()
//   return mn
// }

async function run() {
    const rosNode = await rosnodejs.initNode('/posenet');
    // ROS function for simple recieveing node param
    const getParam = async function(key, default_value){
        if(await rosNode.hasParam(key)){
            const param = await rosNode.getParam(key);
            return param;
        }
        return default_value;
    }
    // Find if GPU is enabled and start tf
    const gpu = await getParam('~gpu', false);
    console.log(gpu);
    if (gpu)
        require('@tensorflow/tfjs-node-gpu');
    else
        require('@tensorflow/tfjs-node');
    const posenet = require('@tensorflow-models/posenet');
    // lowest quality first
    const multiplier = await getParam('~multiplier', 0.5);
    
    // MobileNet image classification
    // const model = await loadModel("/home/jar78/Downloads/MobileNet/model.json");

    // This step requires internet connection as weights are loaded from google servers...
    // TODO download them offline
    const net  = await posenet.load(multiplier);
    // Local variables for sync with ROS
    let buffer = [];
    let newBuffer = false;
    let image_width = 0;
    let image_height = 0;
    let header = null;
    // Parameters for posenet
    const imageScaleFactor = await getParam('~image_scale_factor', 0.5);
    const flipHorizontal = await getParam('~flip_horizontal', false);
    const outputStride = await getParam('~output_stride', 16);
    const maxPoseDetections = await getParam('~max_pose', 5);
    const scoreThreshold = await getParam('~score_threshold', 0.5);
    const nmsRadius = await getParam('~nms_radius', 20);
    const multiPerson = await getParam('~multiPerson', false);
    // topic names
    const camera_topic = await getParam('~topic','/camera/image_raw');
    const output_topic = await getParam('~poses_topic','/poses');
    // ROS topics
    let pub = rosNode.advertise(output_topic, PosesMsg);
    //
    let sub = rosNode.subscribe(camera_topic, sensor_msgs.Image,
        (data) => {
            // TODO more encodings
            if (data.encoding == 'bgr8'){
                // Change the encoding to rgb8 
                // Atm not implemented, this means red is blue and vice versa which leads to worse results
                // data.data = swapChannels(data.data);
                data.encoding = 'rgb8';
            }
            // Currently works only with rgb8 data
            assert(data.encoding == 'rgb8');
            buffer = data.data;
            newBuffer = true;
            header = data.header;
            image_height = data.height;
            image_width = data.width;
        }
    );
    // Loop for detecting poses
    const DetectingPoses = async function (){
    if (newBuffer == false)  return;
        let tensor = tf.tensor3d(buffer, [image_height,image_width,3], 'int32');
        newBuffer = false;
        let pose_msg = new pose_msgs.Poses()
        if (multiPerson === true) {
             const poses = await net.estimateMultiplePoses(tensor, imageScaleFactor, flipHorizontal, outputStride,
                                                           maxPoseDetections, scoreThreshold,nmsRadius);
            for (let i = 0; i < poses.length; i++){
                pose_msg.poses.push(new pose_msgs.Pose());
                pose_msg.poses[i]["score"] = poses[i]["score"];
                for (let k = 0; k < poses[i]["keypoints"].length; k++){
                    pose_msg.poses[i].keypoints.push(new pose_msgs.Keypoint());
                    pose_msg.poses[i].keypoints[k].score = poses[i]["keypoints"][k]["score"];
                    pose_msg.poses[i].keypoints[k].part = poses[i]["keypoints"][k]["part"];
                    pose_msg.poses[i].keypoints[k].position.x = poses[i]["keypoints"][k]["position"]["x"];
                    pose_msg.poses[i].keypoints[k].position.y = poses[i]["keypoints"][k]["position"]["y"];

                }
            }           
        }
        else{
            const poses = await net.estimateSinglePose(tensor, imageScaleFactor, flipHorizontal, outputStride);
            pose_msg.poses.push(new pose_msgs.Pose());
            let i = 0;
            pose_msg.poses[i]["score"] = poses["score"];
            for (let k = 0; k < poses["keypoints"].length; k++){
                pose_msg.poses[i].keypoints.push(new pose_msgs.Keypoint());
                pose_msg.poses[i].keypoints[k].score = poses["keypoints"][k]["score"];
                pose_msg.poses[i].keypoints[k].part = poses["keypoints"][k]["part"];
                pose_msg.poses[i].keypoints[k].position.x = poses["keypoints"][k]["position"]["x"];
                pose_msg.poses[i].keypoints[k].position.y = poses["keypoints"][k]["position"]["y"];
            }
        }

        // MobileNet Image Classification
        // const predictions = await model.classify(tensor);
        // console.log('Predictions: ');
        // console.log(predictions);

        tensor.dispose();

        pub.publish(pose_msg);
    }
    setInterval(DetectingPoses, 10);
}





run();
