/**
 * Teacher Dashboard & Student Registration handler.
 */
let regWebcam = null;
let regCaptureInterval = null;
let capturedFramesCount = 0;
const TOTAL_REQUIRED_FRAMES = 50;

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialise Registration Form bindings
    const regForm = document.getElementById('student-registration-form');
    const startCaptureBtn = document.getElementById('btn-start-capture');
    
    if (regForm && startCaptureBtn) {
        startCaptureBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            if (validateRegistrationForm()) {
                await initiateRegistration();
            }
        });
    }

    // 2. Initialise Analytics Charts if Chart.js is loaded and container exists
    const analyticsChartCanvas = document.getElementById('analytics-trend-chart');
    if (analyticsChartCanvas && typeof Chart !== 'undefined') {
        renderAnalyticsChart(analyticsChartCanvas);
    }
});

function validateRegistrationForm() {
    const fields = ['reg-roll', 'reg-name', 'reg-branch', 'reg-section', 'reg-semester', 'reg-email', 'reg-password'];
    for (let f of fields) {
        const el = document.getElementById(f);
        if (!el || !el.value.trim()) {
            alert(`Please enter a valid value for ${el ? el.previousElementSibling.textContent : f}`);
            return false;
        }
    }
    return true;
}

async function initiateRegistration() {
    const startCaptureBtn = document.getElementById('btn-start-capture');
    const regFormCard = document.getElementById('registration-fields-card');
    const webcamCard = document.getElementById('registration-webcam-card');
    
    const rollNo = document.getElementById('reg-roll').value.trim();
    const name = document.getElementById('reg-name').value.trim();
    const branch = document.getElementById('reg-branch').value.trim();
    const section = document.getElementById('reg-section').value.trim();
    const semester = document.getElementById('reg-semester').value.trim();
    const email = document.getElementById('reg-email').value.trim();
    const password = document.getElementById('reg-password').value.trim();

    startCaptureBtn.disabled = true;
    startCaptureBtn.textContent = "Processing Details...";

    try {
        const response = await fetch('/teacher/api/register/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                roll_no: rollNo,
                name: name,
                branch: branch,
                section: section,
                semester: semester,
                email: email,
                password: password
            })
        });

        const result = await response.json();
        if (result.success) {
            // Swap viewports to webcam capture mode
            regFormCard.style.display = 'none';
            webcamCard.style.display = 'block';
            
            // Start Webcam
            const video = document.getElementById('reg-webcam-video');
            regWebcam = new WebcamHelper(video);
            await regWebcam.start();
            
            // Begin continuous capturing
            startCaptureLoop(rollNo);
        } else {
            alert(result.message || "Registration failed to start.");
            startCaptureBtn.disabled = false;
            startCaptureBtn.textContent = "Continue to Face Capture";
        }
    } catch (err) {
        console.error("Init registration error:", err);
        startCaptureBtn.disabled = false;
        startCaptureBtn.textContent = "Continue to Face Capture";
    }
}

function startCaptureLoop(rollNo) {
    capturedFramesCount = 0;
    const progressFill = document.getElementById('capture-progress-fill');
    const progressText = document.getElementById('capture-progress-text');
    const promptText = document.getElementById('capture-direction-prompt');
    
    regCaptureInterval = setInterval(async () => {
        if (capturedFramesCount >= TOTAL_REQUIRED_FRAMES) {
            clearInterval(regCaptureInterval);
            await completeRegistration(rollNo);
            return;
        }

        // Set Face ID direction guides
        updateOnscreenPrompt(capturedFramesCount, promptText);

        const frame = regWebcam.getFrameBase64(0.8);
        if (!frame) return;

        try {
            const response = await fetch('/teacher/api/register/upload_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    roll_no: rollNo,
                    image: frame
                })
            });

            const result = await response.json();
            if (result.success) {
                capturedFramesCount = result.count;
                
                // Update Progress bar percentage
                const pct = Math.round((capturedFramesCount / TOTAL_REQUIRED_FRAMES) * 100);
                progressFill.style.width = `${pct}%`;
                progressText.textContent = `Captured: ${capturedFramesCount}/${TOTAL_REQUIRED_FRAMES} frames (${pct}%)`;
            } else {
                promptText.textContent = result.message || "Align face to webcam...";
            }
        } catch (e) {
            console.error("Frame capture upload error:", e);
        }
    }, 400); // capture frame every 400ms to allow client pose change
}

function updateOnscreenPrompt(count, element) {
    if (count < 15) {
        element.textContent = "Look straight at the webcam (Front Profile)...";
        element.style.color = '#6366f1';
    } else if (count < 30) {
        element.textContent = "Turn your head SLOWLY to the LEFT...";
        element.style.color = '#0ea5e9';
    } else if (count < 45) {
        element.textContent = "Turn your head SLOWLY to the RIGHT...";
        element.style.color = '#0ea5e9';
    } else {
        element.textContent = "Tilt your head slightly UP and DOWN...";
        element.style.color = '#10b981';
    }
}

async function completeRegistration(rollNo) {
    const promptText = document.getElementById('capture-direction-prompt');
    const progressFill = document.getElementById('capture-progress-fill');
    
    promptText.textContent = "Compiling face embeddings. Please wait...";
    promptText.style.color = '#f59e0b';
    
    if (regWebcam) regWebcam.stop();

    try {
        const response = await fetch('/teacher/api/register/complete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ roll_no: rollNo })
        });

        const result = await response.json();
        if (result.success) {
            promptText.textContent = "Registration Completed Successfully!";
            promptText.style.color = '#10b981';
            
            setTimeout(() => {
                window.location.href = '/teacher/dashboard';
            }, 1500);
        } else {
            alert(result.message || "Failed to compile registration embeddings.");
            resetRegistrationUI();
        }
    } catch (e) {
        console.error("Complete registration error:", e);
        resetRegistrationUI();
    }
}

function resetRegistrationUI() {
    if (regWebcam) regWebcam.stop();
    document.getElementById('registration-fields-card').style.display = 'block';
    document.getElementById('registration-webcam-card').style.display = 'none';
    const startCaptureBtn = document.getElementById('btn-start-capture');
    startCaptureBtn.disabled = false;
    startCaptureBtn.textContent = "Continue to Face Capture";
}

function renderAnalyticsChart(canvasEl) {
    // Generate analytics trend using chart.js
    const ctx = canvasEl.getContext('2d');
    
    // Draw nice mock gradient
    const primaryGlow = ctx.createLinearGradient(0, 0, 0, 400);
    primaryGlow.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
    primaryGlow.addColorStop(1, 'rgba(99, 102, 241, 0.0)');
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Average Attendance Trend (%)',
                data: [78, 82, 85, 80, 89, 92],
                borderColor: '#6366f1',
                backgroundColor: primaryGlow,
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' },
                    min: 50,
                    max: 100
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}
