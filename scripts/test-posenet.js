const rosnodejs = require('rosnodejs');
const sensor_msgs = rosnodejs.require('sensor_msgs').msg;
const StringMsg = rosnodejs.require('std_msgs').msg.String;

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');

const cv = require('opencv4nodejs');

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
        let img = undefined;
        
        if(imgData.encoding == "rgb8")
            img = imgData.data;
        else if(imgData.encoding == "bgr8")
            img = new cv.Mat(Buffer.from(imgData.data), imgData.height, imgData.width, cv.CV_8UC3).cvtColor(cv.COLOR_BGR2RGB);

        //cv.imshow('test', img);
        //cv.waitKey(1);

        if(img != undefined){
            console.time("posenet")
            let tensor = tf.tensor3d(img.getDataAsArray(), [imgData.height,imgData.width,3], 'int32');
            // net.estimateSinglePose(tensor, paramScaleFactor, paramFlipHorizontal, paramOutputStride)
            // .then((pose) => {
            //     if(pose['score'] > 0.2){
            //         for(let k = 0; k < pose['keypoints'].length; k++){
            //             if(pose['keypoints'][k]['score'] > 0.2)
            //                 img.drawCircle(new cv.Point(pose['keypoints'][k]['position']['x'], pose['keypoints'][k]['position']['y']),
            //                 4, new cv.Vec3(255, 0, 0), 2, 8, 0);
            //         }
            //     }
            //     cv.imshow('test', img.cvtColor(cv.COLOR_RGB2BGR));
            //     cv.waitKey(1);
            // });

            pose = await net.estimateSinglePose(tensor, paramScaleFactor, paramFlipHorizontal, paramOutputStride);
            if(pose['score'] > 0.2){
                for(let k = 0; k < pose['keypoints'].length; k++){
                    if(pose['keypoints'][k]['score'] > 0.2)
                        img.drawCircle(new cv.Point(pose['keypoints'][k]['position']['x'], pose['keypoints'][k]['position']['y']),
                        4, new cv.Vec3(255, 0, 0), 2, 8, 0);
                }
            }
            console.timeEnd("posenet");

            cv.imshow('test', img.cvtColor(cv.COLOR_RGB2BGR));
            cv.waitKey(1);  
            
        } else {
            rosnodejs.log.warning('Unknown image format, skipping');
        }
    }, options);

}

if (require.main === module)
    main();