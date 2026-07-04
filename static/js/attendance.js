/**
 * Live Face Recognition & Attendance Tracking Loop
 */
let webcam = null;
let detectionInterval = null;
let isSessionRunning = false;
const clientSessionId = 'client_' + Math.random().toString(36).substr(2, 9);

document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('canvas-overlay');
    const startBtn = document.getElementById('btn-start-session');
    const pauseBtn = document.getElementById('btn-pause-session');
    const endBtn = document.getElementById('btn-end-session');
    const subjectTitle = document.getElementById('active-subject');
    
    if (!video || !canvas) return;

    webcam = new WebcamHelper(video, canvas);

    // Bind Session Control buttons
    if (startBtn) {
        startBtn.addEventListener('click', async () => {
            await startDetectionSession();
        });
    }

    if (pauseBtn) {
        pauseBtn.addEventListener('click', () => {
            pauseDetectionSession();
        });
    }

    if (endBtn) {
        endBtn.addEventListener('click', async () => {
            await endDetectionSession();
        });
    }
});

async function startDetectionSession() {
    try {
        await webcam.start();
        isSessionRunning = true;
        
        // Update button states
        document.getElementById('btn-start-session').style.display = 'none';
        document.getElementById('btn-pause-session').style.display = 'inline-flex';
        document.getElementById('btn-end-session').disabled = false;
        
        // Reset blink detector on backend
        await fetch('/attendance/api/reset_liveness', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ client_id: clientSessionId })
        });
        
        // Start loop
        startProcessingLoop();
        showStatusNotification("Camera active. Starting face recognition...", "info");
    } catch (err) {
        console.error("Session start error:", err);
    }
}

function pauseDetectionSession() {
    isSessionRunning = false;
    if (detectionInterval) {
        clearInterval(detectionInterval);
    }
    webcam.stop();
    
    // Clear canvas overlay
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    document.getElementById('btn-start-session').style.display = 'inline-flex';
    document.getElementById('btn-pause-session').style.display = 'none';
    showStatusNotification("Attendance tracking paused.", "warning");
}

async function endDetectionSession() {
    isSessionRunning = false;
    if (detectionInterval) {
        clearInterval(detectionInterval);
    }
    webcam.stop();
    
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    try {
        const response = await fetch('/teacher/api/end_session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const result = await response.json();
        
        if (result.success) {
            showStatusNotification("Attendance session completed and saved.", "success");
            setTimeout(() => {
                window.location.href = '/teacher/dashboard';
            }, 1500);
        } else {
            alert(result.message || "Failed to end session.");
        }
    } catch (err) {
        console.error("End session error:", err);
    }
}

function startProcessingLoop() {
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas.getContext('2d');
    
    // Run detection every 300ms
    detectionInterval = setInterval(async () => {
        if (!isSessionRunning) return;
        
        const frameBase64 = webcam.getFrameBase64(0.7);
        if (!frameBase64) return;
        
        try {
            const response = await fetch('/attendance/api/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: frameBase64,
                    client_id: clientSessionId
                })
            });
            
            const data = await response.json();
            
            // Clear previous overlays
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (data.success && data.faces_count > 0 && data.box) {
                drawBoundingBox(ctx, data);
                updateLivenessIndicator(data.liveness);
                
                if (data.recognized && data.attendance) {
                    handleAttendanceResult(data);
                }
            } else {
                updateLivenessIndicator({ is_live: false, details: { method: "searching", info: "Scanning..." } });
            }
            
        } catch (error) {
            console.error("Error processing frame:", error);
        }
    }, 300);
}

function drawBoundingBox(ctx, data) {
    const [x, y, w, h] = data.box;
    const isLive = data.liveness && data.liveness.is_live;
    const isRecognized = data.recognized;
    
    // Set color: green if recognized & live, yellow if checking, red if unknown
    let color = '#ef4444'; // default red (unknown)
    if (isRecognized) {
        color = isLive ? '#10b981' : '#f59e0b'; // green if fully verified, orange if liveness pending
    }
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.shadowBlur = 10;
    ctx.shadowColor = color;
    
    // Draw modern bracket corners instead of full rectangle
    const offset = 15;
    
    // Top-Left Corner
    ctx.beginPath();
    ctx.moveTo(x, y + offset);
    ctx.lineTo(x, y);
    ctx.lineTo(x + offset, y);
    ctx.stroke();
    
    // Top-Right Corner
    ctx.beginPath();
    ctx.moveTo(x + w - offset, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + offset);
    ctx.stroke();
    
    // Bottom-Left Corner
    ctx.beginPath();
    ctx.moveTo(x, y + h - offset);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + offset, y + h);
    ctx.stroke();
    
    // Bottom-Right Corner
    ctx.beginPath();
    ctx.moveTo(x + w - offset, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - offset);
    ctx.stroke();
    
    // Label tag
    ctx.shadowBlur = 0;
    ctx.fillStyle = color;
    ctx.fillRect(x, y - 30, w, 30);
    
    // Text inside tag
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 13px Inter';
    let text = isRecognized ? `${data.name} (${data.confidence}%)` : "Unknown Face";
    if (isRecognized && !isLive) {
        text += " - Blink Required";
    }
    ctx.fillText(text, x + 8, y - 10);
}

function updateLivenessIndicator(liveness) {
    const indicator = document.getElementById('liveness-method');
    const eyeState = document.getElementById('liveness-ear');
    const indicatorDot = document.getElementById('indicator-dot');
    
    if (!indicator || !liveness) return;
    
    const details = liveness.details || {};
    indicator.textContent = `Method: ${details.method || 'estimating'}`;
    
    if (details.eye_state) {
        eyeState.textContent = `Status: Eyes ${details.eye_state} (EAR: ${details.ear})`;
    } else if (details.movement_score) {
        eyeState.textContent = `Movement Score: ${details.movement_score}`;
    } else {
        eyeState.textContent = details.info || 'Scanning liveness...';
    }
    
    if (liveness.is_live) {
        indicatorDot.className = 'status-dot active';
    } else {
        indicatorDot.className = 'status-dot blink';
    }
}

let lastMarkedStudent = null;
let lastMarkedTimestamp = 0;

function handleAttendanceResult(data) {
    const rollNo = data.roll_no;
    const name = data.name;
    const attStatus = data.attendance.status;
    const timeMarked = data.attendance.time;
    
    // Limit double logging trigger alerts
    const now = Date.now();
    if (lastMarkedStudent === rollNo && (now - lastMarkedTimestamp) < 8000) {
        return; // Don't spam notifications
    }
    
    lastMarkedStudent = rollNo;
    lastMarkedTimestamp = now;
    
    if (data.attendance.success) {
        // Trigger visual success notification
        showToast(name, attStatus, timeMarked);
        
        // Append row to the logging table
        appendLogsTable(rollNo, name, timeMarked, attStatus);
        
        // Refresh Session Stats dynamically
        updateDashboardStats();
    }
}

function appendLogsTable(rollNo, name, time, status) {
    const tbody = document.getElementById('realtime-logs-body');
    if (!tbody) return;
    
    // Check if student already in the table list to avoid UI duplicates
    const existingRow = document.getElementById(`row-${rollNo}`);
    if (existingRow) return;
    
    const tr = document.createElement('tr');
    tr.id = `row-${rollNo}`;
    tr.className = 'animate-fade-in';
    
    tr.innerHTML = `
        <td><img src="/static/uploads/profile_photos/${rollNo}_${name}.jpg" class="face-thumbnail" onerror="this.src='/static/images/default_profile.png'"></td>
        <td><b>${rollNo}</b></td>
        <td>${name}</td>
        <td>${time}</td>
        <td><span class="badge badge-success">${status}</span></td>
    `;
    
    // Insert at top of logs
    tbody.insertBefore(tr, tbody.firstChild);
}

async function updateDashboardStats() {
    try {
        const response = await fetch('/teacher/api/stats');
        const stats = await response.json();
        
        // Update stats DOM counts
        document.getElementById('stat-present').textContent = stats.present;
        document.getElementById('stat-absent').textContent = stats.absent;
        document.getElementById('stat-percentage').textContent = `${stats.percentage}%`;
    } catch (err) {
        console.error("Error loading stats:", err);
    }
}

function showToast(name, status, time) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = 'custom-toast';
    
    // Play text-to-speech greeting if possible
    try {
        const greeting = new SpeechSynthesisUtterance(`Welcome ${name}. Attendance ${status}.`);
        window.speechSynthesis.speak(greeting);
    } catch (e) {}
    
    toast.innerHTML = `
        <div style="font-size: 1.5rem; color: #10b981;">✓</div>
        <div>
            <h4 style="font-size: 0.95rem; color: white;">Attendance Verified</h4>
            <p style="font-size: 0.8rem; color: #94a3b8;">${name} marked <b>${status}</b> at ${time}</p>
        </div>
    `;
    
    container.appendChild(toast);
    
    // Remove toast after 4s
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s reverse forwards';
        setTimeout(() => {
            container.removeChild(toast);
        }, 300);
    }, 4000);
}

function showStatusNotification(message, type = 'info') {
    // Basic status notice logger
    console.log(`[Status ${type}]: ${message}`);
}
