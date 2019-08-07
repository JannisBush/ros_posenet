const rosnodejs = require('rosnodejs');
const sensor_msgs = rosnodejs.require('sensor_msgs').msg;
const StringMsg = rosnodejs.require('std_msgs').msg.String;

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');

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

function debugView(imgData, pose) {
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
    let paramName = '/posenet';
    let paramImgNode = '/image_raw';
    let paramScaleFactor = 0.25;
    let paramFlipHorizontal = false;
    let paramOutputStride = 16;

    rosnodejs.log.info('Starting ROS node.');
    const rosNode = await rosnodejs.initNode(paramName)
    rosnodejs.log.info('Node ' + paramName + ' registered. Loading PoseNet.');

    // net = await posenet.load({
    //     architecture: guiState.model.architecture,
    //     outputStride: guiState.model.outputStride,
    //     inputResolution: guiState.model.inputResolution,
    //     multiplier: guiState.model.multiplier,
    //     quantBytes: guiState.model.quantBytes,
    //   });

    const net = await posenet.load();
    rosnodejs.log.info('PoseNet model loaded.');

    let options = {queueSize: 1, throttleMs: 100};
    const imgSub = rosNode.subscribe(paramImgNode, sensor_msgs.Image, async (imgData) => {
        const imgCanvas = formatImage(imgData);
        console.time("posenet")
        pose = await net.estimateSinglePose(imgCanvas, paramScaleFactor, paramFlipHorizontal, paramOutputStride);
        console.timeEnd("posenet");
        debugView(imgData, pose);
    }, options);

}

if (require.main === module)
    main();