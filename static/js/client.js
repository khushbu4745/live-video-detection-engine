const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvasElement');
const ctx = canvas.getContext('2d');

const taskTabs = document.querySelectorAll('.tab');
const modelSelect = document.getElementById('modelSelect');
const zeroShotControls = document.getElementById('zeroShotControls');
const updateBtn = document.getElementById('updateBtn');
const classInput = document.getElementById('classInput');

let ws;
let currentPredictions = [];
let streamInterval;
let currentTask = "detection";
let currentModel = "standard";

const engineRegistry = {
    "detection": [
        { value: "standard", label: "YOLO11 (Standard)" },
        { value: "rtdetr", label: "RT-DETR (Transformer)" },
        { value: "world", label: "YOLO-World (Zero-Shot)" },
        { value: "owlv2", label: "OWLv2 (Zero-Shot)" }
    ],
    "segmentation": [
        { value: "standard", label: "YOLO11 Segmentation" }
    ],
    "pose": [
        { value: "standard", label: "YOLO11 Pose Estimation" }
    ],
    "obb": [
        { value: "standard", label: "YOLO11 Oriented Boxes" }
    ],
    "classification": [
        { value: "standard", label: "YOLO11 Classification" }
    ]
};

const skeletonConnections = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], 
    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
];

function getClassColor(label, alpha = 1.0) {
    let hash = 0;
    for (let i = 0; i < label.length; i++) {
        hash = label.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash % 360);
    return `hsla(${hue}, 80%, 60%, ${alpha})`;
}

function initWebSocket() {
    ws = new WebSocket(`ws://${location.host}/ws`);
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        currentPredictions = data.boxes || data || []; 
    };

    ws.onclose = () => {
        console.log("WebSocket closed. Stopping stream.");
        if (streamInterval) clearInterval(streamInterval);
    };
}

navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
    .then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = () => video.play();
    })
    .catch(err => alert("Camera access failed: " + err.message));

video.addEventListener('playing', () => {
    if (streamInterval) return; 
    
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    
    initWebSocket();
    
    streamInterval = setInterval(() => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(canvas.toDataURL('image/jpeg', 0.8));
        }

        currentPredictions.forEach(item => {
            const type = item.task_type;
            const labelText = `${item.label} (${(item.conf * 100).toFixed(1)}%)`;

            const strokeColor = getClassColor(item.label, 1.0);
            const fillColor = getClassColor(item.label, 0.3);

            ctx.fillStyle = strokeColor;
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 2;
            ctx.font = 'bold 16px Arial';

            if (type === 'detection') {
                ctx.strokeRect(item.x1, item.y1, item.x2 - item.x1, item.y2 - item.y1);
                ctx.fillText(labelText, item.x1, item.y1 > 20 ? item.y1 - 8 : item.y1 + 20);
            } 
            else if (type === 'segmentation' && item.polygon && item.polygon.length > 0) {
                ctx.beginPath();
                ctx.moveTo(item.polygon[0][0], item.polygon[0][1]);
                for (let i = 1; i < item.polygon.length; i++) {
                    ctx.lineTo(item.polygon[i][0], item.polygon[i][1]);
                }
                ctx.closePath();
                
                ctx.fillStyle = fillColor;
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = strokeColor;
                const topPoint = item.polygon.reduce((min, pt) => pt[1] < min[1] ? pt : min, item.polygon[0]);
                ctx.fillText(labelText, topPoint[0], topPoint[1] > 20 ? topPoint[1] - 8 : topPoint[1] + 20);
            } 
            else if (type === 'pose' && item.keypoints) {
                ctx.beginPath();
                skeletonConnections.forEach(([i, j]) => {
                    const [x1, y1] = item.keypoints[i] || [0, 0];
                    const [x2, y2] = item.keypoints[j] || [0, 0];
                    if (x1 && x2) {
                        ctx.moveTo(x1, y1);
                        ctx.lineTo(x2, y2);
                    }
                });
                ctx.stroke();

                item.keypoints.forEach(([x, y]) => {
                    if (x) {
                        ctx.beginPath();
                        ctx.arc(x, y, 4, 0, Math.PI * 2);
                        ctx.fill();
                    }
                });

                const validPoints = item.keypoints.filter(pt => pt[0] !== 0 && pt[1] !== 0);
                if (validPoints.length > 0) {
                    const topPoint = validPoints.reduce((min, pt) => pt[1] < min[1] ? pt : min, validPoints[0]);
                    ctx.fillStyle = strokeColor;
                    ctx.fillText(labelText, topPoint[0], topPoint[1] > 20 ? topPoint[1] - 12 : topPoint[1] + 20);
                }
            } 
            else if (type === 'obb' && item.corners && item.corners.length === 4) {
                ctx.beginPath();
                ctx.moveTo(item.corners[0][0], item.corners[0][1]);
                for (let i = 1; i < 4; i++) {
                    ctx.lineTo(item.corners[i][0], item.corners[i][1]);
                }
                ctx.closePath();
                ctx.stroke();
                ctx.fillText(labelText, item.corners[0][0], item.corners[0][1] - 8);
            } 
            else if (type === 'classification') {
                ctx.font = 'bold 28px Arial';
                ctx.fillText(`Global Match: ${labelText}`, 20, 40);
            }
        });
    }, 100);
}, { once: true });

function dispatchPipelineSwitch() {
    currentPredictions = [];
    
    if (currentModel === "world" || currentModel === "owlv2") {
        zeroShotControls.style.display = "flex";
    } else {
        zeroShotControls.style.display = "none";
    }
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ 
            type: "switch_pipeline", 
            task: currentTask,
            model: currentModel 
        }));
    }
}

taskTabs.forEach(tab => {
    tab.addEventListener('click', (e) => {
        taskTabs.forEach(t => t.classList.remove('active'));
        e.target.classList.add('active');
        
        currentTask = e.target.dataset.task;
        
        modelSelect.innerHTML = '';
        engineRegistry[currentTask].forEach(model => {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.label;
            modelSelect.appendChild(option);
        });
        
        currentModel = modelSelect.value;
        dispatchPipelineSwitch();
    });
});

modelSelect.addEventListener('change', (e) => {
    currentModel = e.target.value;
    dispatchPipelineSwitch();
});

updateBtn.addEventListener('click', () => {
    const customClasses = classInput.value.split(',').map(item => item.trim()).filter(i => i);
    if (ws && ws.readyState === WebSocket.OPEN && customClasses.length > 0) {
        currentPredictions = [];
        ws.send(JSON.stringify({
            type: "update_classes",
            classes: customClasses
        }));
    }
});