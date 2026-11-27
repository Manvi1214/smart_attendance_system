/**
 * Live Face Recognition & Attendance Tracking Loop
 * Streamlined Single-Blink Active Challenge & AI-based Passive PAD
 */
let webcam = null;
let isSessionRunning = false;
let isProcessingFrame = false;
let loopTimeoutId = null;
const clientSessionId = 'client_' + Math.random().toString(36).substr(2, 9);

document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('canvas-overlay');
    const startBtn = document.getElementById('btn-start-session');
    const pauseBtn = document.getElementById('btn-pause-session');
    const endBtn = document.getElementById('btn-end-session');
    
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
        isProcessingFrame = false;
        
        // Update button states
        const startBtn = document.getElementById('btn-start-session');
        const pauseBtn = document.getElementById('btn-pause-session');
        const endBtn = document.getElementById('btn-end-session');
        
        if (startBtn) startBtn.style.display = 'none';
        if (pauseBtn) pauseBtn.style.display = 'inline-flex';
        if (endBtn) endBtn.disabled = false;
        
        // Reset challenge session on backend
        try {
            await fetch('/attendance/api/reset_liveness', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ client_id: clientSessionId })
            });
        } catch (e) {
            console.warn("Liveness reset warning:", e);
        }
        
        // Start non-blocking processing loop
        processNextFrame();
        updateLivenessUI({ status: "Detecting Face...", message: "Scanning camera feed..." });
    } catch (err) {
        console.error("Session start error:", err);
    }
}

function pauseDetectionSession() {
    isSessionRunning = false;
    isProcessingFrame = false;
    if (loopTimeoutId) {
        clearTimeout(loopTimeoutId);
        loopTimeoutId = null;
    }
    if (webcam) webcam.stop();
    
    // Clear canvas overlay
    const canvas = document.getElementById('canvas-overlay');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    const startBtn = document.getElementById('btn-start-session');
    const pauseBtn = document.getElementById('btn-pause-session');
    if (startBtn) startBtn.style.display = 'inline-flex';
    if (pauseBtn) pauseBtn.style.display = 'none';
    
    updateLivenessUI({ status: "Detecting Face...", message: "Attendance tracking paused." });
}

async function endDetectionSession() {
    isSessionRunning = false;
    isProcessingFrame = false;
    if (loopTimeoutId) {
        clearTimeout(loopTimeoutId);
        loopTimeoutId = null;
    }
    if (webcam) webcam.stop();
    
    const canvas = document.getElementById('canvas-overlay');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    try {
        const response = await fetch('/teacher/api/end_session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const result = await response.json();
        
        if (result.success) {
            updateLivenessUI({ status: "Attendance Marked Successfully", message: "Session saved." });
            setTimeout(() => {
                window.location.href = '/teacher/dashboard';
            }, 1200);
        } else {
            alert(result.message || "Failed to end session.");
        }
    } catch (err) {
        console.error("End session error:", err);
    }
}

/**
 * High-speed asynchronous frame capture loop.
 * Uses requestAnimationFrame + isProcessingFrame lock to eliminate frame piling.
 */
async function processNextFrame() {
    if (!isSessionRunning) return;
    
    if (isProcessingFrame) {
        loopTimeoutId = setTimeout(processNextFrame, 60);
        return;
    }
    
    const frameData = webcam.getOptimizedFrame(0.65, 480);
    if (!frameData || !frameData.base64) {
        loopTimeoutId = setTimeout(processNextFrame, 100);
        return;
    }
    
    isProcessingFrame = true;
    const canvas = document.getElementById('canvas-overlay');
    const ctx = canvas ? canvas.getContext('2d') : null;
    
    try {
        const response = await fetch('/attendance/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: frameData.base64,
                client_id: clientSessionId
            })
        });
        
        const data = await response.json();
        
        if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        
        if (data && data.success && data.faces_count > 0 && data.box && ctx) {
            // Re-scale bounding box coordinates to canvas display resolution
            const invScale = 1.0 / (frameData.scale || 1.0);
            const scaledBox = [
                Math.round(data.box[0] * invScale),
                Math.round(data.box[1] * invScale),
                Math.round(data.box[2] * invScale),
                Math.round(data.box[3] * invScale)
            ];
            
            drawBoundingBox(ctx, data, scaledBox);
            updateLivenessUI(data);
            
            // Only handle attendance if actively marked and verified
            if (data.recognized && data.attendance) {
                handleAttendanceResult(data);
            }
        } else {
            updateLivenessUI({
                status: (data && data.status) ? data.status : "Detecting Face...",
                message: "Detecting Face... Please look directly into the camera."
            });
        }
        
    } catch (error) {
        console.error("Frame recognition error:", error);
    } finally {
        isProcessingFrame = false;
        if (isSessionRunning) {
            loopTimeoutId = setTimeout(() => {
                requestAnimationFrame(processNextFrame);
            }, 10);
        }
    }
}

/**
 * Draws modern sci-fi HUD bracket corners around the face and a high-contrast label tag.
 * Corrects horizontal canvas mirror scaling so text remains normally readable.
 */
function drawBoundingBox(ctx, data, box) {
    const [x, y, w, h] = box;
    const status = data.status || "";
    const isLive = data.liveness && data.liveness.is_live;
    const isSpoof = (status === "Spoofing Detected");
    const isFailed = (status === "Verification Failed");
    
    // Determine HUD color scheme
    let color = '#38bdf8'; // Default cyan (Detecting/Face Detected)
    if (isSpoof || isFailed) {
        color = '#ef4444'; // Red (Presentation Attack / Verification Failed)
    } else if (status === "Attendance Marked Successfully" || isLive || status === "Real Person Verified" || status === "Blink Verified") {
        color = '#10b981'; // Emerald Green (Real Person Verified / Blink Verified)
    } else if (status === "Please Blink Once" || status === "Checking Anti-Spoofing...") {
        color = '#f59e0b'; // Amber / Orange (Verification in progress)
    }
    
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.shadowBlur = 14;
    ctx.shadowColor = color;
    
    // Draw modern bracket corners
    const offset = Math.min(20, Math.floor(w * 0.2));
    
    // Top-Left
    ctx.beginPath();
    ctx.moveTo(x, y + offset);
    ctx.lineTo(x, y);
    ctx.lineTo(x + offset, y);
    ctx.stroke();
    
    // Top-Right
    ctx.beginPath();
    ctx.moveTo(x + w - offset, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + offset);
    ctx.stroke();
    
    // Bottom-Left
    ctx.beginPath();
    ctx.moveTo(x, y + h - offset);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + offset, y + h);
    ctx.stroke();
    
    // Bottom-Right
    ctx.beginPath();
    ctx.moveTo(x + w - offset, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - offset);
    ctx.stroke();
    
    // Draw text banner background
    ctx.shadowBlur = 0;
    const tagHeight = 28;
    ctx.fillStyle = color;
    
    // Rounded label top tag
    const radius = 6;
    const tagY = Math.max(0, y - tagHeight - 4);
    ctx.beginPath();
    ctx.moveTo(x + radius, tagY);
    ctx.lineTo(x + w - radius, tagY);
    ctx.quadraticCurveTo(x + w, tagY, x + w, tagY + radius);
    ctx.lineTo(x + w, tagY + tagHeight);
    ctx.lineTo(x, tagY + tagHeight);
    ctx.lineTo(x, tagY + radius);
    ctx.quadraticCurveTo(x, tagY, x + radius, tagY);
    ctx.closePath();
    ctx.fill();
    
    // Construct label text based on exact verification state
    let label = "";
    const liv = (data.liveness && data.liveness.details) ? data.liveness.details : {};
    const timeLeft = liv.time_remaining ? ` (${liv.time_remaining}s)` : "";
    
    if (isSpoof) {
        label = "⚠️ SPOOFING DETECTED";
    } else if (isFailed) {
        label = "❌ Verification Failed - Retrying";
    } else if (status === "Attendance Marked Successfully") {
        label = `✓ ${data.name || 'Student'} Marked Present`;
    } else if (status === "Real Person Verified") {
        if (data.attendance && data.attendance.already_marked) {
            label = `✓ ${data.name || 'Student'} (Already Marked)`;
        } else if (data.name && data.name !== "Unknown") {
            label = `✓ ${data.name} Verified`;
        } else {
            label = `✓ Real Person Verified`;
        }
    } else if (status === "Checking Anti-Spoofing...") {
        label = "🛡️ Checking Anti-Spoofing...";
    } else if (status === "Blink Verified") {
        label = "✓ Blink Verified";
    } else if (status === "Please Blink Once") {
        label = `👁️ Please Blink Once${timeLeft}`;
    } else if (status === "Face Detected") {
        label = "Face Detected";
    } else {
        label = "Detecting Face...";
    }
    
    // Flip text horizontally because parent canvas has CSS `transform: scaleX(-1)`
    ctx.save();
    ctx.translate(x + w / 2, tagY + tagHeight / 2 + 4);
    ctx.scale(-1, 1);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, 0, 0);
    ctx.restore();
    
    ctx.restore();
}

/**
 * Updates the liveness badge and displays the exact required status messages:
 * - Detecting Face...
 * - Face Detected
 * - Please Blink Once
 * - Blink Verified
 * - Checking Anti-Spoofing...
 * - Real Person Verified
 * - Spoofing Detected
 * - Verification Failed
 * - Attendance Marked Successfully
 */
function updateLivenessUI(data) {
    const boxEl = document.getElementById('liveness-status-box');
    const titleEl = document.getElementById('liveness-status-title');
    const dotEl = document.getElementById('indicator-dot');
    const methodEl = document.getElementById('liveness-method');
    const instructionEl = document.getElementById('liveness-ear');
    
    const badgeBlink = document.getElementById('step-badge-blink');
    const badgePad = document.getElementById('step-badge-pad');
    
    if (!titleEl || !dotEl) return;
    
    const status = (data && data.status) ? data.status : "Detecting Face...";
    titleEl.textContent = status;
    
    // Reset status box classes
    if (boxEl) {
        boxEl.classList.remove('status-spoof', 'status-verified', 'status-verifying', 'status-failed');
    }
    
    const liv = (data.liveness && data.liveness.details) ? data.liveness.details : {};
    
    switch (status) {
        case "Spoofing Detected":
            dotEl.className = 'status-dot danger blink';
            if (boxEl) boxEl.classList.add('status-spoof');
            if (methodEl) methodEl.textContent = "Presentation Attack Detected";
            if (instructionEl) instructionEl.textContent = liv.diagnosis || "Photograph or screen display rejected!";
            if (badgeBlink) badgeBlink.className = 'step-badge completed';
            if (badgePad) badgePad.className = 'step-badge spoof';
            break;
            
        case "Verification Failed":
            dotEl.className = 'status-dot danger blink';
            if (boxEl) boxEl.classList.add('status-failed');
            if (methodEl) methodEl.textContent = "Blink Timeout";
            if (instructionEl) instructionEl.textContent = liv.message || "Blink not detected. Retrying naturally...";
            if (badgeBlink) badgeBlink.className = 'step-badge failed';
            if (badgePad) badgePad.className = 'step-badge pending';
            break;
            
        case "Attendance Marked Successfully":
            dotEl.className = 'status-dot active';
            if (boxEl) boxEl.classList.add('status-verified');
            if (methodEl) methodEl.textContent = "Attendance Recorded ✓";
            if (instructionEl) instructionEl.textContent = `${data.name || 'Student'} marked Present`;
            if (badgeBlink) badgeBlink.className = 'step-badge completed';
            if (badgePad) badgePad.className = 'step-badge completed';
            break;
            
        case "Real Person Verified":
            dotEl.className = 'status-dot active';
            if (boxEl) boxEl.classList.add('status-verified');
            if (methodEl) methodEl.textContent = "Liveness Confirmed ✓";
            if (instructionEl) instructionEl.textContent = "Authentic person verified. Matching facial embedding...";
            if (badgeBlink) badgeBlink.className = 'step-badge completed';
            if (badgePad) badgePad.className = 'step-badge completed';
            break;
            
        case "Checking Anti-Spoofing...":
            dotEl.className = 'status-dot warning blink';
            if (boxEl) boxEl.classList.add('status-verifying');
            if (methodEl) methodEl.textContent = "Passive PAD Scan";
            if (instructionEl) instructionEl.textContent = "Analyzing skin texture and screen reflection...";
            if (badgeBlink) badgeBlink.className = 'step-badge completed';
            if (badgePad) badgePad.className = 'step-badge in_progress';
            break;
            
        case "Blink Verified":
            dotEl.className = 'status-dot active';
            if (boxEl) boxEl.classList.add('status-verified');
            if (methodEl) methodEl.textContent = "Eye Blink Confirmed ✓";
            if (instructionEl) instructionEl.textContent = "Blink verified. Checking presentation attack detection...";
            if (badgeBlink) badgeBlink.className = 'step-badge completed';
            if (badgePad) badgePad.className = 'step-badge in_progress';
            break;
            
        case "Please Blink Once":
            dotEl.className = 'status-dot warning blink';
            if (boxEl) boxEl.classList.add('status-verifying');
            const timeTxt = liv.time_remaining ? ` (${liv.time_remaining}s remaining)` : "";
            if (methodEl) methodEl.textContent = "Active Liveness Check";
            if (instructionEl) instructionEl.textContent = `Please blink your eyes naturally once${timeTxt}`;
            if (badgeBlink) badgeBlink.className = 'step-badge in_progress';
            if (badgePad) badgePad.className = 'step-badge pending';
            break;
            
        case "Face Detected":
            dotEl.className = 'status-dot info';
            if (methodEl) methodEl.textContent = "Face Positioned";
            if (instructionEl) instructionEl.textContent = "Calibrating eye aspect ratio (EAR)...";
            if (badgeBlink) badgeBlink.className = 'step-badge in_progress';
            if (badgePad) badgePad.className = 'step-badge pending';
            break;
            
        case "Detecting Face...":
        default:
            dotEl.className = 'status-dot info blink';
            if (methodEl) methodEl.textContent = "Single-Blink + Passive PAD";
            if (instructionEl) instructionEl.textContent = data.message || "Looking for student face...";
            if (badgeBlink) badgeBlink.className = 'step-badge pending';
            if (badgePad) badgePad.className = 'step-badge pending';
            break;
    }
}

let lastMarkedStudent = null;
let lastMarkedTimestamp = 0;

function handleAttendanceResult(data) {
    const rollNo = data.roll_no;
    const name = data.name;
    const attData = data.attendance || {};
    const attStatus = attData.status || "Present";
    const timeMarked = attData.time || new Date().toLocaleTimeString();
    
    // Prevent repeated popup alerts for the same student within 8 seconds
    const now = Date.now();
    if (lastMarkedStudent === rollNo && (now - lastMarkedTimestamp) < 8000) {
        return;
    }
    
    if (attData.success) {
        lastMarkedStudent = rollNo;
        lastMarkedTimestamp = now;
        
        // Show visual toast notification
        showToast(name, attStatus, timeMarked);
        
        // Append row to the live logging table
        appendLogsTable(rollNo, name, timeMarked, attStatus);
        
        // Refresh Session Stats dynamically
        updateDashboardStats();
    }
}

function appendLogsTable(rollNo, name, time, status) {
    const tbody = document.getElementById('realtime-logs-body');
    if (!tbody) return;
    
    // Check if student already in the table list
    const existingRow = document.getElementById(`row-${rollNo}`);
    if (existingRow) return;
    
    // Clear "No markings" placeholder if present
    if (tbody.children.length === 1 && tbody.children[0].cells.length === 1) {
        tbody.innerHTML = '';
    }
    
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
        
        const pres = document.getElementById('stat-present');
        const abs = document.getElementById('stat-absent');
        const pct = document.getElementById('stat-percentage');
        
        if (pres) pres.textContent = stats.present;
        if (abs) abs.textContent = stats.absent;
        if (pct) pct.textContent = `${stats.percentage}%`;
    } catch (err) {
        console.error("Error loading stats:", err);
    }
}

function showToast(name, status, time) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = 'custom-toast';
    
    const isAlready = (status === "Already Marked");
    const speechMsg = isAlready ? `Welcome ${name}. You are already marked present.` : `Welcome ${name}. Attendance marked.`;
    const subText = isAlready ? `${name} is already marked <b>Present</b> at ${time}` : `${name} marked <b>${status}</b> at ${time}`;
    
    // Play text-to-speech confirmation if enabled
    try {
        const greeting = new SpeechSynthesisUtterance(speechMsg);
        greeting.rate = 1.0;
        window.speechSynthesis.speak(greeting);
    } catch (e) {}
    
    toast.innerHTML = `
        <div style="font-size: 1.5rem; color: #10b981;">✓</div>
        <div>
            <h4 style="font-size: 0.95rem; color: white;">${isAlready ? 'Attendance Recorded' : 'Attendance Verified'}</h4>
            <p style="font-size: 0.8rem; color: #94a3b8;">${subText}</p>
        </div>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s reverse forwards';
        setTimeout(() => {
            if (container.contains(toast)) {
                container.removeChild(toast);
            }
        }, 300);
    }, 4000);
}
