#!/usr/bin/env node

/**
 * ROS PoseNet
 * 
 * This is the main script of the `posenet` package. It reads a ROS color camera 
 * stream (such as `/image_raw`) and then uses PoseNet to detect the skeleton
 * (pose) of a single or multiple users in the image.
 * 
 * The current script supports the two PoseNet models, MobileNetV1 and ResNet50,
 * as well as the single and multi-pose algorithms.
 */

// Packages required by ROS.
const rosnodejs = require('rosnodejs');
const sensor_msgs = rosnodejs.require('sensor_msgs').msg;
const pose_msgs = rosnodejs.require('ros_posenet').msg;

// Packages required by PoseNet. The PoseNet package itself is loaded in the
// main function, as it requires the CPU or the GPU tensorflow library to be
// loaded first, and that depends on the configuration read from the launch
// file.
const tf = require('@tensorflow/tfjs');
const cv = require('opencv4nodejs');
const { createImageData, createCanvas } = require('canvas')


/**
 * Transforms a ROS image message into a Canvas object, so that it can be
 * inputted into the PoseNet neural network.
 * @param {sensor_msgs.Image} imgData A ROS image message.
 * @returns {Canvas} A Canvas object with the same content as the ROS image.
 */
function formatImage(imgData){
    // Converts the original color mode to RGBA. 
    let conversionCode = null;

    if(imgData.encoding == "rgb8")
        conversionCode = cv.COLOR_RGB2RGBA;
    else if(imgData.encoding == "bgr8")
        conversionCode = cv.COLOR_BGR2RGBA;
    else
        throw "Unknown image format.";

    let img = new cv.Mat(Buffer.from(imgData.data), imgData.height, 
                imgData.width, cv.CV_8UC3).cvtColor(conversionCode);
    
    // Creates the Canvas object, draw and return it.
    const imgCanvas = createCanvas(imgData.height, imgData.width);
    const imgCtx = imgCanvas.getContext('2d');
    let tempImg = createImageData(
        new Uint8ClampedArray(img.getData()),
        imgData.width,
        imgData.height
    );
    imgCtx.putImageData(tempImg, 0, 0);
    return imgCanvas;
}


/**
 * Draws the detected keypoints over the input image and shows to the user.
 * 
 * Used for debugging only.
 * @param {sensor_msgs.Image} imgData A ROS image message.
 * @param {Pose} poses The poses detected by PoseNet.
 */
function debugView (imgData, poses) {
    img = new cv.Mat(Buffer.from(imgData.data), imgData.height, 
            imgData.width, cv.CV_8UC3).cvtColor(cv.COLOR_BGR2RGBA);
    
    poses.forEach( pose => {
        if(pose['score'] > 0.2){
            pose['keypoints'].forEach(keypoint => {
                if(keypoint['score'] > 0.2)
                    img.drawCircle(new cv.Point(
                        keypoint['position']['x'], 
                        keypoint['position']['y']),
                    4, new cv.Vec3(129, 245, 60), 2, 8, 0);
            });
        }
    });

    cv.imshow('test', img.cvtColor(cv.COLOR_RGB2BGR));
    cv.waitKey(1);
}


/**
 * Provides the `/posenet` node and output topic.
 * 
 * This function provides the `/posenet` node. It subscribes to an image topic
 * (e.g. `/image_raw`) and feed it to PoseNet to obtain the estimated poses.
 * The estimated poses are then published to an output topic (e.g. `/poses`).
 */
async function main() {
    // Register node with ros; `rosNode` is used to load the parameters.
    const rosNode = await rosnodejs.initNode("/posenet")
    rosnodejs.log.info('Node /posenet registered.');
    
    // Load all parameters from `posenet.launch`.
    const paramImgTopic = await getParam('/posenet/image_topic', '/image_raw');
    const paramPosesTopic = await getParam('/posenet/poses_topic', '/poses');
    const paramGPU = await getParam('/posenet/gpu', false);
    const paramArchitecture = await getParam('/posenet/architecture', 'MobileNetV1');
    const paramMultiplier = await getParam('/posenet/multiplier', 0.5);
    const paramInputResolution = await getParam('/posenet/input_resolution', 257);
    const paramQuantBytes = await getParam('/posenet/quant_bytes', 4)
    const paramOutputStride = await getParam('/posenet/output_stride', 16);
    const paramFlipHorizontal = await getParam('/posenet/flip_horizontal', false);
    const paramMultiPose = await getParam('/posenet/multi_pose', false);
    const paramMaxDetection = await getParam('/posenet/max_detection', 5);
    const paramMinPoseConf = await getParam('/posenet/min_pose_confidence', 0.1);
    const paramMinPartConf = await getParam('/posenet/min_part_confidence', 0.5);
    const paramNmsRadius = await getParam('/posenet/nms_radius', 30);

    // Load PoseNet dependencies and model.
    if (paramGPU)
        require('@tensorflow/tfjs-node-gpu');
    else
        require('@tensorflow/tfjs-node');
    const posenet = require('@tensorflow-models/posenet');
        
    const net = await posenet.load({
        architecture: paramArchitecture,
        outputStride: paramOutputStride,
        inputResolution: paramInputResolution,
        multiplier: paramMultiplier,
        quantBytes: paramQuantBytes,
    });

    rosnodejs.log.info('PoseNet model loaded.');

    // Creates the publishing topic and subscribe to the image topic.
    let posePub = rosNode.advertise(paramPosesTopic, pose_msgs.Poses);
    let options = {queueSize: 1, throttleMs: 100};
    let imgSub;
    if (paramMultiPose)
        imgSub = rosNode.subscribe(paramImgTopic, sensor_msgs.Image, 
            multiPoseCallback, options);
    else
        imgSub = rosNode.subscribe(paramImgTopic, sensor_msgs.Image, 
            singlePoseCallback, options);
    
    // Main function ends here. Bellow you can find the utility functions and
    // callbacks.

    /**
     * ROS function reading parameters from the parameters server.
     * @param {String} key The parameter's name as per the launch file.
     * @param {*} default_value The default value that should be loaded in case
     *                          it is not provided.
     * @returns The value for the given parameter.
     */
    async function getParam (key, default_value){
        if(await rosNode.hasParam(key)){
            const param = await rosNode.getParam(key);
            return param;
        }
        rosnodejs.log.warn('Parameter ' + key +
            ' not found; using default value: ' + default_value);
        return default_value;
    }
    

    /**
     * Callback for the pose detection when a single pose is considered.
     * 
     * This callback process the input ROS image using PoseNet to detect a
     * single pose. The result is published into the output topic.
     * @param {sensors_msgs.Image} imgData A ROS image message.
     */
    async function singlePoseCallback(imgData){
        const imgCanvas = formatImage(imgData);
        console.time("posenet")
        pose = await net.estimateSinglePose(imgCanvas, 
            {flipHorizontal: paramFlipHorizontal});
        console.timeEnd("posenet");
        posePub.publish(buildOutputMessage([pose]));
        debugView(imgData, [pose]);
    }


    /**
     * Callback for the pose detection when multiple poses are considered.
     * 
     * This callback process the input ROS image using PoseNet to detect
     * multiple poses. The result is published into the output topic.
     * @param {sensors_msgs.Image} imgData A ROS image message.
     */
    async function multiPoseCallback(imgData){
        const imgCanvas = formatImage(imgData);
        console.time("posenet")
        poses = await net.estimateMultiplePoses(imgCanvas, {
            flipHorizontal: paramFlipHorizontal,
            maxDetections: paramMaxDetection,
            scoreThreshold: paramMinPartConf,
            nmsRadius: paramNmsRadius
        });
        console.timeEnd("posenet");
        posePub.publish(buildOutputMessage(poses));
        debugView(imgData, poses);
    }


    /**
     * Converts a Pose object into a ROS message.
     * @param {[Poses]} poses The poses outputted by PoseNet.
     * @returns {pose_msgs.Poses} The ROS message that will be published.
     */
    function buildOutputMessage(poses) {
        let msg = new pose_msgs.Poses();
        poses.forEach(poseData => {
            if (poseData['score'] > paramMinPoseConf) {
                pose = new pose_msgs.Pose();
                pose.score = poseData['score'];
                poseData['keypoints'].forEach(keypointData => {
                    keypoint = new pose_msgs.Keypoint();
                    keypoint.score = keypointData['score'];
                    keypoint.part = keypointData['part'];
                    keypoint.position.x = keypointData['position']['x'];
                    keypoint.position.y = keypointData['position']['y'];
                    pose.keypoints.push(keypoint);
                });
                msg.poses.push(pose);
            }
        });
        return msg;
    }
}


// Executes the main function.
if (require.main === module)
    main();
