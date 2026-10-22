
export const CANVAS_WIDTH=640;
export const CANVAS_HEIGHT=480;
export const TOOLBAR_HEIGHT=90;

export const TOOLS=Object.freeze({
    DRAW:"draw",
    ERASER:"eraser",
    SPRAY:"spray",
    CRAYON:"crayon",
    COLOR:"color",
    SIZE:"size"
});

export const PALETTE=Object.freeze([
    "#ff0000",
    "#ffa500",
    "#ffff00",
    "#00ff00",
    "#00ffff",
    "#0000ff",
    "#ff00ff",
    "#ffffff"
]);

const INITIAL_COLOR="#ffb450";
const ZOOM_MIN=0.6;
const ZOOM_MAX=2.5;
const ZOOM_SENSITIVITY=0.0025;

function clamp(value,min,max){
    return Math.max(min,Math.min(max,value));
}

function randomInteger(min,max){
    return Math.floor(Math.random()*(max-min+1))+min;
}

function colorToName(color){
    const names={
        "#ff0000":"RED",
        "#ffa500":"ORANGE",
        "#ffff00":"YELLOW",
        "#00ff00":"GREEN",
        "#00ffff":"CYAN",
        "#0000ff":"BLUE",
        "#ff00ff":"MAGENTA",
        "#ffffff":"WHITE",
        "#ffb450":"ORANGE"
    };
    return names[color.toLowerCase()]||"CUSTOM";
}

export class DrawingEngine {
    constructor(canvas){
        if(!(canvas instanceof HTMLCanvasElement)){
            throw new Error("DrawingEngine requires an HTML canvas.");
        }

        this.canvas=canvas;
        this.canvas.width=CANVAS_WIDTH;
        this.canvas.height=CANVAS_HEIGHT;
        this.ctx=canvas.getContext("2d");

        if(!this.ctx){
            throw new Error("Could not create drawing canvas context.");
        }

        this.layer=document.createElement("canvas");
        this.layer.width=CANVAS_WIDTH;
        this.layer.height=CANVAS_HEIGHT;
        this.layerCtx=this.layer.getContext("2d",{
            willReadFrequently:true
        });

        this.tool=TOOLS.DRAW;
        this.drawColor=INITIAL_COLOR;
        this.brushThickness=8;
        this.eraserThickness=35;
        this.zoom=1.0;
        this.previousZoomDistance=0;
        this.previousPoint=null;
        this.showSizeSlider=false;
        this.lastPoint=null;
        this.lastGesture="none";
        this.onChange=null;
        this.onSaveRequested=null;
        this.saveSelectionLatched=false;

        this.render();
    }

    getState(){
        return {
            tool:this.tool,
            color:this.drawColor,
            colorName:colorToName(this.drawColor),
            brushSize:this.brushThickness,
            zoom:this.zoom,
            showSizeSlider:this.showSizeSlider
        };
    }

    notifyChange(){
        if(typeof this.onChange==="function"){
            this.onChange(this.getState());
        }
    }

    drawCircle(ctx,x,y,radius,color){
        ctx.beginPath();
        ctx.arc(x,y,Math.max(1,radius),0,Math.PI*2);
        ctx.fillStyle=color;
        ctx.fill();
    }

    drawLine(ctx,x1,y1,x2,y2,width,color){
        ctx.beginPath();
        ctx.moveTo(x1,y1);
        ctx.lineTo(x2,y2);
        ctx.strokeStyle=color;
        ctx.lineWidth=width;
        ctx.lineCap="round";
        ctx.lineJoin="round";
        ctx.stroke();
    }

    getZoomViewport(){
        const newWidth=Math.trunc(CANVAS_WIDTH/this.zoom);
        const newHeight=Math.trunc(CANVAS_HEIGHT/this.zoom);
        const centerX=CANVAS_WIDTH/2;
        const centerY=CANVAS_HEIGHT/2;

        const startX=Math.max(
            0,
            Math.trunc(centerX-Math.trunc(newWidth/2))
        );

        const startY=Math.max(
            0,
            Math.trunc(centerY-Math.trunc(newHeight/2))
        );

        const endX=Math.min(CANVAS_WIDTH,startX+newWidth);
        const endY=Math.min(CANVAS_HEIGHT,startY+newHeight);

        return {
            x:startX,
            y:startY,
            width:endX-startX,
            height:endY-startY,
            mappingWidth:newWidth,
            mappingHeight:newHeight
        };
    }

    screenToCanvas(point){
        const viewport=this.getZoomViewport();

        return {
            x:Math.trunc(
                viewport.x+
                (point.x/CANVAS_WIDTH)*viewport.mappingWidth
            ),
            y:Math.trunc(
                viewport.y+
                (point.y/CANVAS_HEIGHT)*viewport.mappingHeight
            )
        };
    }

    drawPen(point){
        if(!this.previousPoint){
            this.previousPoint={...point};
        }

        const steps=8;

        for(let i=0;i<=steps;i++){
            const x=Math.trunc(
                this.previousPoint.x+
                (point.x-this.previousPoint.x)*i/steps
            );

            const y=Math.trunc(
                this.previousPoint.y+
                (point.y-this.previousPoint.y)*i/steps
            );

            this.drawCircle(
                this.layerCtx,
                x,
                y,
                Math.max(1,Math.trunc(this.brushThickness/2)),
                this.drawColor
            );
        }

        this.previousPoint={...point};
    }

    drawEraser(point){
        if(!this.previousPoint){
            this.previousPoint={...point};
        }

        this.layerCtx.save();
        this.layerCtx.globalCompositeOperation="destination-out";

        this.drawLine(
            this.layerCtx,
            this.previousPoint.x,
            this.previousPoint.y,
            point.x,
            point.y,
            this.eraserThickness,
            "#ffffff"
        );

        this.layerCtx.restore();
        this.previousPoint={...point};
    }

    drawSpray(point){
        this.previousPoint=null;

        for(let i=0;i<25;i++){
            const offsetX=randomInteger(
                -this.brushThickness,
                this.brushThickness
            );

            const offsetY=randomInteger(
                -this.brushThickness,
                this.brushThickness
            );

            const x=point.x+offsetX;
            const y=point.y+offsetY;

            if(
                x>=0&&
                x<CANVAS_WIDTH&&
                y>=0&&
                y<CANVAS_HEIGHT
            ){
                this.drawCircle(
                    this.layerCtx,
                    x,
                    y,
                    1,
                    this.drawColor
                );
            }
        }
    }

    drawCrayon(point){
        if(!this.previousPoint){
            this.previousPoint={...point};
        }

        for(let i=0;i<5;i++){
            const x1=this.previousPoint.x+randomInteger(-3,3);
            const y1=this.previousPoint.y+randomInteger(-3,3);
            const x2=point.x+randomInteger(-3,3);
            const y2=point.y+randomInteger(-3,3);

            this.drawLine(
                this.layerCtx,
                x1,
                y1,
                x2,
                y2,
                Math.max(1,Math.trunc(this.brushThickness/3)),
                this.drawColor
            );
        }

        this.previousPoint={...point};
    }

    selectToolbarTool(point){
        if(point.y>=TOOLBAR_HEIGHT){
            this.saveSelectionLatched=false;
            return;
        }

        const section=Math.trunc(point.x/91);
        const tools=[
            TOOLS.DRAW,
            TOOLS.ERASER,
            TOOLS.SPRAY,
            TOOLS.CRAYON,
            TOOLS.COLOR,
            TOOLS.SIZE
        ];

        if(section>=0&&section<=5){
            this.tool=tools[section];
            this.showSizeSlider=this.tool===TOOLS.SIZE;
            this.saveSelectionLatched=false;
            this.notifyChange();
        }else if(section===6&&!this.saveSelectionLatched){
            this.saveSelectionLatched=true;

            if(typeof this.onSaveRequested==="function"){
                this.onSaveRequested();
            }

            this.tool=TOOLS.DRAW;
            this.showSizeSlider=false;
            this.notifyChange();
        }
    }

    handleColorPalette(point){
        if(this.tool!==TOOLS.COLOR){
            return;
        }

        const startX=120;
        const startY=TOOLBAR_HEIGHT+20;
        const boxSize=45;
        const gap=8;

        for(let i=0;i<PALETTE.length;i++){
            const x=startX+i*(boxSize+gap);
            const y=startY;

            if(
                point.x>x&&
                point.x<x+boxSize&&
                point.y>y&&
                point.y<y+boxSize
            ){
                this.drawColor=PALETTE[i];
                this.tool=TOOLS.DRAW;
                this.showSizeSlider=false;
                this.notifyChange();
                break;
            }
        }
    }

    handleSizeSlider(point,frame){
        if(this.tool!==TOOLS.SIZE||!frame.indexUp){
            return;
        }

        const top=120;
        const bottom=390;

        if(
            point.x>550&&
            point.x<640&&
            point.y>top&&
            point.y<bottom
        ){
            const ratio=(bottom-point.y)/(bottom-top);
            this.brushThickness=Math.trunc(2+ratio*28);
            this.brushThickness=clamp(this.brushThickness,2,30);
            this.notifyChange();
        }
    }

    handleZoom(frame){
        const distance=frame.zoomDistance;

        if(this.previousZoomDistance!==0){
            const delta=distance-this.previousZoomDistance;

            if(Math.abs(delta)>4){
                this.zoom+=delta*ZOOM_SENSITIVITY;
                this.zoom=clamp(this.zoom,ZOOM_MIN,ZOOM_MAX);
                this.notifyChange();
            }
        }

        this.previousZoomDistance=distance;
    }

    handleTracking(frame){
        if(!frame?.detected||!frame.point){
            this.previousPoint=null;
            this.previousZoomDistance=0;
            this.lastPoint=null;
            this.lastGesture="none";
            this.saveSelectionLatched=false;
            this.render();
            return;
        }

        const point=frame.point;
        this.lastPoint=point;
        this.lastGesture=frame.gesture;

        const indexUp=frame.indexUp;
        const middleUp=frame.middleUp;
        const ringUp=frame.ringUp;

        if(
            indexUp&&
            middleUp&&
            ringUp&&
            point.y>TOOLBAR_HEIGHT+20
        ){
            this.handleZoom(frame);
        }else{
            this.previousZoomDistance=0;
        }

        if(indexUp&&middleUp&&!ringUp){
            this.previousPoint=null;

            if(point.y<TOOLBAR_HEIGHT){
                this.selectToolbarTool(point);
            }else{
                this.saveSelectionLatched=false;
            }

            this.handleColorPalette(point);
        }

        if(this.showSizeSlider){
            this.handleSizeSlider(point,frame);
        }else if(
            indexUp&&
            !middleUp&&
            this.tool!==TOOLS.SIZE
        ){
            if(point.y>TOOLBAR_HEIGHT){
                const canvasPoint=this.screenToCanvas(point);

                switch(this.tool){
                    case TOOLS.DRAW:
                        this.drawPen(canvasPoint);
                        break;

                    case TOOLS.ERASER:
                        this.drawEraser(canvasPoint);
                        break;

                    case TOOLS.SPRAY:
                        this.drawSpray(canvasPoint);
                        break;

                    case TOOLS.CRAYON:
                        this.drawCrayon(canvasPoint);
                        break;
                }
            }else{
                this.previousPoint=null;
            }
        }else{
            this.previousPoint=null;
        }

        this.render();
    }

    render(){
        const ctx=this.ctx;

        ctx.clearRect(
            0,
            0,
            CANVAS_WIDTH,
            CANVAS_HEIGHT
        );

        const viewport=this.getZoomViewport();

        ctx.drawImage(
            this.layer,
            viewport.x,
            viewport.y,
            viewport.width,
            viewport.height,
            0,
            0,
            CANVAS_WIDTH,
            CANVAS_HEIGHT
        );

        if(this.tool===TOOLS.COLOR){
            const startX=120;
            const startY=TOOLBAR_HEIGHT+20;

            for(let i=0;i<PALETTE.length;i++){
                const x=startX+i*53;

                ctx.fillStyle=PALETTE[i];
                ctx.fillRect(x,startY,45,45);

                ctx.strokeStyle="#e6e6e6";
                ctx.lineWidth=2;
                ctx.strokeRect(x,startY,45,45);
            }
        }

        if(this.showSizeSlider){
            const sliderX=600;
            const top=120;
            const bottom=390;

            ctx.fillStyle="#1e1e2a";
            ctx.fillRect(
                570,
                top-20,
                60,
                bottom-top+40
            );

            this.drawLine(
                ctx,
                sliderX,
                top,
                sliderX,
                bottom,
                5,
                "#b4b4be"
            );

            const knobY=bottom-
                ((this.brushThickness-2)/28)*
                (bottom-top);

            this.drawCircle(
                ctx,
                sliderX,
                knobY,
                15,
                "#00ffb4"
            );

            ctx.font="bold 16px Arial";
            ctx.fillStyle="#e6e6e6";
            ctx.fillText("SIZE",570,top-35);
        }

        if(this.lastPoint){
            const point=this.lastPoint;

            ctx.beginPath();
            ctx.arc(
                point.x,
                point.y,
                18,
                0,
                Math.PI*2
            );

            ctx.strokeStyle=this.drawColor;
            ctx.lineWidth=2;
            ctx.stroke();

            this.drawCircle(
                ctx,
                point.x,
                point.y,
                7,
                this.drawColor
            );
        }
    }

    clear(){
        this.layerCtx.clearRect(
            0,
            0,
            CANVAS_WIDTH,
            CANVAS_HEIGHT
        );

        this.previousPoint=null;
        this.previousZoomDistance=0;
        this.zoom=1.0;
        this.tool=TOOLS.DRAW;
        this.showSizeSlider=false;

        this.notifyChange();
        this.render();
    }

    getDrawingCanvas(){
        return this.layer;
    }

    getImageData(){
        return this.layerCtx.getImageData(
            0,
            0,
            CANVAS_WIDTH,
            CANVAS_HEIGHT
        );
    }

    hasDrawing(){
        const data=this.getImageData().data;

        for(let i=3;i<data.length;i+=4){
            if(data[i]>0){
                return true;
            }
        }

        return false;
    }

    downloadPNG(filename="aircanvas.png"){
        const exportCanvas=document.createElement("canvas");
        exportCanvas.width=CANVAS_WIDTH;
        exportCanvas.height=CANVAS_HEIGHT;

        const exportCtx=exportCanvas.getContext("2d");

        exportCtx.fillStyle="#121219";
        exportCtx.fillRect(
            0,
            0,
            CANVAS_WIDTH,
            CANVAS_HEIGHT
        );

        exportCtx.drawImage(this.layer,0,0);

        const link=document.createElement("a");
        link.download=filename;
        link.href=exportCanvas.toDataURL("image/png");
        link.click();
    }
}
