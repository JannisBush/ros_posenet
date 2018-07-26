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


async function run() {
    const rosNode = await rosnodejs.initNode('/posenet')
    // ROS function for simple recieveing node param
    const getParam = async function(key, default_value){
        if(await rosNode.hasParam(key)){
            const param = await rosNode.getParam(key)
            return param
        }
        return default_value
    }
    // Find if GPU is enabled and start tf
    const gpu = await getParam('~gpu', false)
    console.log(gpu)
    if (gpu)
        require('@tensorflow/tfjs-node-gpu');
    else
        require('@tensorflow/tfjs-node');
    const posenet = require('@tensorflow-models/posenet')
    // lowest quality first
    const multiplier = await getParam('~multiplier', 0.5)
    // This step requires internet connection as weights are loaded from google servers...
    // TODO download them offline
    const net  = await posenet.load(multiplier);
    // Local variables for sync with ROS
    let buffer = []
    let newBuffer = false
    let image_width = 0
    let image_height = 0
    let header = null
    // Parameters for posenet
    const imageScaleFactor = await getParam('~image_scale_factor', 0.5);
    const flipHorizontal = await getParam('~flip_horizontal', false);
    const outputStride = await getParam('~output_stride', 16);
    const maxPoseDetections = await getParam('~max_pose', 5);
    const scoreThreshold = await getParam('~score_threshold', 0.5);
    const nmsRadius = await getParam('~nms_radius', 20);
    // topic names
    const camera_topic = await getParam('~topic','/camera/image_raw')
    const output_topic = await getParam('~poses_topic','/poses')
    // ROS topics
    let pub = rosNode.advertise(output_topic, StringMsg)
    //
    let sub = rosNode.subscribe(camera_topic, sensor_msgs.Image,
        (data) => {
            // TODO more encodings
            // Currently works wonly with rgb8 data
            assert(data.encoding == 'rgb8')
            buffer = data.data
            newBuffer = true
            header = data.header
            image_height = data.height
            image_width = data.width
        }
    );
    // Loop for detecting poses
    const DetectingPoses = async function (){
    if (newBuffer == false)  return
        let tensor = tf.tensor3d(buffer, [image_height,image_width,3], 'int32')
        newBuffer = false
        const poses = await net.estimateMultiplePoses(tensor, imageScaleFactor, flipHorizontal, outputStride,
                                                       maxPoseDetections, scoreThreshold,nmsRadius);
        tensor.dispose();
        pub.publish({data: stringify(poses)})
    }
    setInterval(DetectingPoses, 10);
}





run();
