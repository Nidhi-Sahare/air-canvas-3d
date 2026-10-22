
import {HandTracker} from "./hand-tracking.js";
import {DrawingEngine} from "./drawing.js";

const video=document.getElementById("webcam");
const canvas=document.getElementById("canvas");
const startCameraButton=document.getElementById("startCamera");
const message=document.getElementById("message");
const toolStatus=document.getElementById("toolStatus");
const sizeStatus=document.getElementById("sizeStatus");
const zoomStatus=document.getElementById("zoomStatus");
const colorStatus=document.getElementById("colorStatus");

const tracker=new HandTracker();
const drawing=new DrawingEngine(canvas);

let cameraStarting=false;
let cameraRunning=false;
let saveCount=0;
let saveDialogOpen=false;

function setMessage(text){
    message.textContent=text;
}

function updateStatusBar(state=drawing.getState()){
    toolStatus.textContent=`TOOL: ${state.tool.toUpperCase()}`;
    sizeStatus.textContent=`SIZE: ${state.brushSize}`;
    zoomStatus.textContent=`ZOOM: ${state.zoom.toFixed(1)}x`;
    colorStatus.textContent=`COLOR: ${state.colorName}`;
}

drawing.onChange=state=>{
    updateStatusBar(state);
};

function createSaveDialog(){
    const existing=document.getElementById("saveDialog");
    if(existing){
        return existing;
    }

    const dialog=document.createElement("dialog");
    dialog.id="saveDialog";

    dialog.style.cssText=`
        width:min(420px,calc(100vw - 32px));
        padding:30px;
        border:1px solid #19e5db;
        border-radius:14px;
        background:#191b24;
        color:#ffffff;
        text-align:center;
        font-family:Arial,sans-serif;
        box-shadow:0 18px 60px rgba(0,0,0,0.55);
    `;

    dialog.innerHTML=`
        <h2 style="
            margin:0 0 14px;
            color:#19e5db;
            font-size:24px;
        ">
            IMAGE SAVED
        </h2>

        <p id="savedFilename" style="
            color:#d8d8df;
            margin-bottom:24px;
        "></p>

        <button
            id="dialogConvert"
            type="button"
            style="
                width:100%;
                padding:12px;
                border:1px solid #19e5db;
                border-radius:8px;
                background:#102c30;
                color:#19e5db;
                font-size:14px;
                cursor:pointer;
                margin-bottom:12px;
            "
        >
            Convert to 3D
        </button>

        <button
            id="dialogContinue"
            type="button"
            style="
                width:100%;
                padding:12px;
                border:1px solid #414450;
                border-radius:8px;
                background:#252833;
                color:#ffffff;
                font-size:14px;
                cursor:pointer;
            "
        >
            Continue Drawing
        </button>

        <p style="
            margin:18px 0 0;
            font-size:12px;
            color:#aeb2bf;
        ">
            Press 3 to convert to 3D
        </p>
    `;

    document.body.appendChild(dialog);

    dialog.addEventListener("close",()=>{
        saveDialogOpen=false;
    });

    document.getElementById("dialogContinue").addEventListener("click",()=>{
        closeSaveDialog();
    });

    document.getElementById("dialogConvert").addEventListener("click",()=>{
        closeSaveDialog();
        request3DConversion();
    });

    return dialog;
}

function openSaveDialog(filename){
    const dialog=createSaveDialog();
    document.getElementById("savedFilename").textContent=filename;

    if(!dialog.open){
        dialog.showModal();
    }

    saveDialogOpen=true;
}

function closeSaveDialog(){
    const dialog=document.getElementById("saveDialog");

    if(dialog?.open){
        dialog.close();
    }

    saveDialogOpen=false;
}

function saveDrawing(){
    if(saveDialogOpen){
        return;
    }

    const filename=`aircanvas_${saveCount}.png`;
    drawing.downloadPNG(filename);
    saveCount++;
    setMessage(`Saved ${filename}`);
    openSaveDialog(filename);
}

drawing.onSaveRequested=saveDrawing;

function request3DConversion(){
    if(!drawing.hasDrawing()){
        setMessage("Draw something before converting to 3D.");
        return;
    }

    setMessage("3D conversion will be connected next.");
}

tracker.onStatus=({message:text,state})=>{
    console.log("[MediaPipe]",state,text);

    if(state==="connected"){
        cameraRunning=true;
        startCameraButton.textContent="Camera Running";
        startCameraButton.disabled=true;
        setMessage("Camera ready. Raise your index finger to draw.");
    }else if(state==="stopped"){
        cameraRunning=false;
        startCameraButton.disabled=false;
        startCameraButton.textContent="Enable Camera";
    }else{
        setMessage(text);
    }
};

tracker.onError=error=>{
    console.error("Air Canvas camera error:",error);
    setMessage(error.message||"Camera or hand tracking failed.");
    cameraRunning=false;
    startCameraButton.disabled=false;
    startCameraButton.textContent="Enable Camera";
};

tracker.onFrame=frame=>{
    if(saveDialogOpen){
        return;
    }

    drawing.handleTracking(frame);
};

async function startCamera(){
    if(cameraStarting||cameraRunning){
        return;
    }

    cameraStarting=true;
    startCameraButton.disabled=true;
    startCameraButton.textContent="Starting Camera...";

    try{
        await tracker.start(video);
        cameraRunning=true;
        startCameraButton.textContent="Camera Running";
        startCameraButton.disabled=true;
    }catch(error){
        console.error("Unable to start Air Canvas:",error);
        cameraRunning=false;
        startCameraButton.disabled=false;
        startCameraButton.textContent="Try Again";
        setMessage(error.message||"Unable to access camera.");
    }finally{
        cameraStarting=false;
    }
}

document.addEventListener("keydown",event=>{
    const key=event.key.toLowerCase();

    if(key==="r"&&!saveDialogOpen){
        drawing.clear();
        setMessage("Canvas reset.");
    }

    if(key==="3"){
        if(saveDialogOpen){
            closeSaveDialog();
        }

        request3DConversion();
    }

    if(key==="escape"&&saveDialogOpen){
        closeSaveDialog();
    }
});

startCameraButton.addEventListener("click",startCamera);

window.addEventListener("pagehide",()=>{
    tracker.dispose();
});

function initializeApp(){
    updateStatusBar();
    setMessage("Camera not started");
    console.log("Air Canvas 3D browser controller ready.");
}

initializeApp();
