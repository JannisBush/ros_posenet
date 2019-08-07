#!/usr/bin/env node

const rosnodejs = require('rosnodejs');
const sensor_msgs = rosnodejs.require('sensor_msgs').msg;
const StringMsg = rosnodejs.require('std_msgs').msg.String;

const tf = require('@tensorflow/tfjs');
// require('@tensorflow/tfjs-node');
// const posenet = require('@tensorflow-models/posenet');

const cv = require('opencv4nodejs');
const { createImageData, createCanvas } = require('canvas')

function formatImage(imgData){
    let img = null;
        
    if(imgData.encoding == "rgb8")
        img = imgData.data;
    else if(imgData.encoding == "bgr8")
        img = new cv.Mat(Buffer.from(imgData.data), imgData.height, imgData.width, cv.CV_8UC3).cvtColor(cv.COLOR_BGR2RGBA);
    
    if (img == null)
        throw "Unknown image format.";
    
    const imgCanvas = createCanvas(640, 480);
    const imgCtx = imgCanvas.getContext('2d');

    let tempImg = createImageData(
        new Uint8ClampedArray(img.getData()),
        imgData.width,
        imgData.height
    );

    imgCanvas.height = imgData.height;
    imgCanvas.width = imgData.width;
    imgCtx.putImageData(tempImg, 0, 0);

    return imgCanvas;
}

function debugView (imgData, pose) {
    img = new cv.Mat(Buffer.from(imgData.data), imgData.height, imgData.width, cv.CV_8UC3).cvtColor(cv.COLOR_BGR2RGBA);
    
    if(pose['score'] > 0.2){
        for(let k = 0; k < pose['keypoints'].length; k++){
            if(pose['keypoints'][k]['score'] > 0.2)
                img.drawCircle(new cv.Point(pose['keypoints'][k]['position']['x'], pose['keypoints'][k]['position']['y']),
                4, new cv.Vec3(255, 0, 0), 2, 8, 0);
        }
    }

    cv.imshow('test', img.cvtColor(cv.COLOR_RGB2BGR));
    cv.waitKey(1);
}

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
    const paramScaleFactor = await getParam('/posenet/flip_horizontal', 1.0);
    const paramFlipHorizontal = await getParam('/posenet/multiplier', false);
    const paramMultiPose = await getParam('/posenet/multi_pose', false);
    const paramMaxPose = await getParam('/posenet/max_pose', 5);
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

    let options = {queueSize: 1, throttleMs: 100};
    const imgSub = rosNode.subscribe(paramImgTopic, sensor_msgs.Image, async (imgData) => {
        const imgCanvas = formatImage(imgData);
        console.time("posenet")
        pose = await net.estimateSinglePose(imgCanvas, paramScaleFactor, paramFlipHorizontal, paramOutputStride);
        console.timeEnd("posenet");
        debugView(imgData, pose);
    }, options);

    // ROS function for simple recieveing node param
    async function getParam (key, default_value){
        if(await rosNode.hasParam(key)){
            const param = await rosNode.getParam(key);
            return param;
        }
        return default_value;
    }
}

if (require.main === module)
    main();