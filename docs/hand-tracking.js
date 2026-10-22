
import {HandLandmarker,FilesetResolver} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/+esm";

export const GESTURES=Object.freeze({
    NONE:"none",
    DRAW:"draw",
    SELECT:"select",
    ZOOM:"zoom"
});

const WIDTH=640;
const HEIGHT=480;
const MODEL_URL="https://storage.googleapis.com/mediapipe-models/"+"hand_landmarker/hand_landmarker/float16/1/"+"hand_landmarker.task";
const WASM_URL="https://cdn.jsdelivr.net/npm/"+"@mediapipe/tasks-vision@0.10.21/wasm";
const BASE_SMOOTHING=0.35;
const REFERENCE_FPS=30;
const TARGET_INTERVAL=1000/30;
const MAX_INTERVAL=1000/15;

export class HandTracker {
    constructor(){
        this.landmarker=null;
        this.initializing=null;
        this.video=null;
        this.stream=null;
        this.running=false;
        this.starting=false;
        this.animationFrameId=null;
        this.lastVideoTime=-1;
        this.lastInferenceTime=0;
        this.lastTrackingTimestamp=0;
        this.smoothX=0;
        this.smoothY=0;
        this.hasSmoothedPoint=false;
        this.inferenceInterval=TARGET_INTERVAL;
        this.averageInferenceMs=0;
        this.processedFrames=0;
        this.onFrame=null;
        this.onStatus=null;
        this.onError=null;
        this.boundLoop=this.processLoop.bind(this);
    }

    reportStatus(message,state="loading"){
        if(typeof this.onStatus==="function"){
            this.onStatus({message,state});
        }
    }

    reportError(error){
        console.error("[Air Canvas]",error);
        if(typeof this.onError==="function"){
            this.onError(error);
        }
    }

    async initialize(){
        if(this.landmarker){
            return;
        }
        if(this.initializing){
            return this.initializing;
        }

        this.initializing=this.createLandmarker();
        try{
            await this.initializing;
        }finally{
            this.initializing=null;
        }
    }

    async createLandmarker(){
        this.reportStatus("Loading MediaPipe...","loading");

        const vision=await FilesetResolver.forVisionTasks(WASM_URL);
        const options={
            runningMode:"VIDEO",
            numHands:1,
            minHandDetectionConfidence:0.5,
            minHandPresenceConfidence:0.5,
            minTrackingConfidence:0.5
        };

        try{
            this.landmarker=await HandLandmarker.createFromOptions(
                vision,{
                    ...options,
                    baseOptions:{
                        modelAssetPath:MODEL_URL,
                        delegate:"GPU"
                    }
                }
            );
            console.log("[Air Canvas] MediaPipe GPU ready");
        }catch(gpuError){
            console.warn(
                "[Air Canvas] GPU unavailable; "+"using CPU",
                gpuError
            );

            this.landmarker=await HandLandmarker.createFromOptions(
                vision,{
                    ...options,
                    baseOptions:{
                        modelAssetPath:MODEL_URL,
                        delegate:"CPU"
                    }
                }
            );
            console.log("[Air Canvas] MediaPipe CPU ready");
        }

        this.reportStatus("Hand tracking ready.","ready");
    }

    async start(videoElement){
        if(this.running||this.starting){
            return;
        }
        if(!(videoElement instanceof HTMLVideoElement)){
            throw new Error("Valid video element required.");
        }
        if(!navigator.mediaDevices?.getUserMedia){
            throw new Error("Camera requires localhost or HTTPS.");
        }

        this.starting=true;
        this.video=videoElement;

        try{
            this.reportStatus("Requesting camera...","loading");

            this.stream=await navigator.mediaDevices.getUserMedia({
                audio:false,
                video:{
                    width:{ideal:WIDTH},
                    height:{ideal:HEIGHT},
                    frameRate:{ideal:30},
                    facingMode:"user"
                }
            });

            this.video.srcObject=this.stream;
            await this.video.play();
            await this.initialize();
            this.resetSmoothing();

            this.lastVideoTime=-1;
            this.lastInferenceTime=0;
            this.lastTrackingTimestamp=0;
            this.inferenceInterval=TARGET_INTERVAL;
            this.averageInferenceMs=0;
            this.processedFrames=0;
            this.running=true;

            this.reportStatus("Camera connected.","connected");
            this.animationFrameId=requestAnimationFrame(this.boundLoop);
        }catch(error){
            this.stop();
            this.reportError(error);
            throw error;
        }finally{
            this.starting=false;
        }
    }

    resetSmoothing(){
        this.smoothX=0;
        this.smoothY=0;
        this.hasSmoothedPoint=false;
    }

    classifyGesture(landmarks){
        if(!landmarks||landmarks.length<21){
            return {
                gesture:GESTURES.NONE,
                indexUp:false,
                middleUp:false,
                ringUp:false
            };
        }

        const indexUp=landmarks[8].y<landmarks[6].y;
        const middleUp=landmarks[12].y<landmarks[10].y;
        const ringUp=landmarks[16].y<landmarks[14].y;
        let gesture=GESTURES.NONE;

        if(indexUp&&middleUp&&ringUp){
            gesture=GESTURES.ZOOM;
        }else if(indexUp&&middleUp&&!ringUp){
            gesture=GESTURES.SELECT;
        }else if(indexUp&&!middleUp){
            gesture=GESTURES.DRAW;
        }

        return {gesture,indexUp,middleUp,ringUp};
    }

    mirrorLandmarks(landmarks){
        return landmarks.map(point=>({
            x:1-point.x,
            y:point.y,
            z:point.z
        }));
    }

    getSmoothedPoint(landmarks,timestamp){
        const rawX=Math.round(landmarks[8].x*WIDTH);
        const rawY=Math.round(landmarks[8].y*HEIGHT);

        if(!this.hasSmoothedPoint){
            this.smoothX=rawX;
            this.smoothY=rawY;
            this.hasSmoothedPoint=true;
        }else{
            const elapsed=this.lastTrackingTimestamp>0
                ?timestamp-this.lastTrackingTimestamp
                :1000/REFERENCE_FPS;

            const dt=Math.max(1,Math.min(elapsed,100));
            const alpha=1-Math.pow(
                1-BASE_SMOOTHING,
                dt/(1000/REFERENCE_FPS)
            );

            this.smoothX+=alpha*(rawX-this.smoothX);
            this.smoothY+=alpha*(rawY-this.smoothY);
        }

        this.lastTrackingTimestamp=timestamp;
        return {
            x:Math.trunc(this.smoothX),
            y:Math.trunc(this.smoothY)
        };
    }

    getZoomDistance(landmarks){
        const thumb=landmarks[4];
        const index=landmarks[8];
        return Math.hypot(
            (index.x-thumb.x)*WIDTH,
            (index.y-thumb.y)*HEIGHT
        );
    }

    emitFrame(frame){
        if(typeof this.onFrame==="function"){
            this.onFrame(frame);
        }
    }

    processResult(result,timestamp){
        const detected=result.landmarks?.[0];

        if(!detected||detected.length<21){
            this.resetSmoothing();
            this.lastTrackingTimestamp=0;

            this.emitFrame({
                detected:false,
                gesture:GESTURES.NONE,
                point:null,
                landmarks:[],
                zoomDistance:0,
                indexUp:false,
                middleUp:false,
                ringUp:false,
                timestamp
            });
            return;
        }

        const landmarks=this.mirrorLandmarks(detected);
        const fingers=this.classifyGesture(landmarks);
        const point=this.getSmoothedPoint(landmarks,timestamp);

        this.emitFrame({
            detected:true,
            gesture:fingers.gesture,
            point,
            rawPoint:{
                x:Math.round(landmarks[8].x*WIDTH),
                y:Math.round(landmarks[8].y*HEIGHT)
            },
            landmarks,
            zoomDistance:this.getZoomDistance(landmarks),
            indexUp:fingers.indexUp,
            middleUp:fingers.middleUp,
            ringUp:fingers.ringUp,
            timestamp
        });
    }

    updatePerformance(inferenceMs){
        if(this.averageInferenceMs===0){
            this.averageInferenceMs=inferenceMs;
        }else{
            this.averageInferenceMs=
                this.averageInferenceMs*0.9+inferenceMs*0.1;
        }

        this.processedFrames++;

        if(this.processedFrames%30===0){
            const recommendedInterval=Math.max(
                TARGET_INTERVAL,
                this.averageInferenceMs*1.2
            );

            this.inferenceInterval=Math.min(
                MAX_INTERVAL,
                recommendedInterval
            );

            console.log(
                "[Air Canvas] Inference:",
                this.averageInferenceMs.toFixed(1),
                "ms | interval:",
                this.inferenceInterval.toFixed(1),
                "ms"
            );
        }
    }

    processLoop(timestamp){
        if(!this.running){
            return;
        }

        try{
            if(
                this.video?.readyState>=2&&
                this.landmarker&&
                this.video.currentTime!==this.lastVideoTime&&
                (
                    this.lastInferenceTime===0||
                    timestamp-this.lastInferenceTime>=this.inferenceInterval
                )
            ){
                this.lastVideoTime=this.video.currentTime;
                this.lastInferenceTime=timestamp;

                const started=performance.now();
                const result=this.landmarker.detectForVideo(
                    this.video,
                    started
                );
                const finished=performance.now();

                this.updatePerformance(finished-started);
                this.processResult(result,finished);
            }
        }catch(error){
            this.reportError(error);
            this.stop();
            return;
        }

        this.animationFrameId=requestAnimationFrame(this.boundLoop);
    }

    stop(){
        this.running=false;

        if(this.animationFrameId!==null){
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId=null;
        }

        if(this.stream){
            for(const track of this.stream.getTracks()){
                track.stop();
            }
            this.stream=null;
        }

        if(this.video){
            this.video.pause();
            this.video.srcObject=null;
        }

        this.lastVideoTime=-1;
        this.lastTrackingTimestamp=0;
        this.resetSmoothing();
        this.reportStatus("Camera stopped.","stopped");
    }

    dispose(){
        this.stop();

        if(this.landmarker){
            this.landmarker.close();
            this.landmarker=null;
        }

        this.video=null;
    }
}
