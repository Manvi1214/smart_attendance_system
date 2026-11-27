/**
 * Shared webcam utility for capturing frames in browser canvas.
 */
class WebcamHelper {
    constructor(videoElement, canvasElement = null) {
        this.video = videoElement;
        this.canvas = canvasElement || document.createElement('canvas');
        this.captureCanvas = document.createElement('canvas'); // Dedicated capture canvas
        this.stream = null;
        this.isActive = false;
    }

    /**
     * Starts the webcam video stream.
     */
    async start() {
        if (this.isActive) return;
        
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            });
            this.video.srcObject = this.stream;
            this.isActive = true;
            return new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    // Set visible overlay canvas size to match video resolution
                    if (this.canvas) {
                        this.canvas.width = this.video.videoWidth || 640;
                        this.canvas.height = this.video.videoHeight || 480;
                    }
                    resolve(true);
                };
            });
        } catch (error) {
            console.error("Error accessing webcam:", error);
            alert("Could not access webcam. Please make sure camera permissions are enabled.");
            throw error;
        }
    }

    /**
     * Stops the webcam stream.
     */
    stop() {
        if (!this.isActive) return;
        
        try {
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }
            this.video.srcObject = null;
            this.isActive = false;
        } catch (error) {
            console.error("Error stopping webcam:", error);
        }
    }

    /**
     * Captures an optimized JPEG frame for face processing.
     * Downscales to maxWidth (default 480px) to drastically reduce transmission latency and CPU usage
     * while preserving detection and recognition precision.
     */
    getOptimizedFrame(quality = 0.65, maxWidth = 480) {
        if (!this.isActive || !this.video.videoWidth) return null;
        
        const origW = this.video.videoWidth;
        const origH = this.video.videoHeight;
        
        let targetW = origW;
        let targetH = origH;
        let scale = 1.0;
        
        if (maxWidth && origW > maxWidth) {
            scale = maxWidth / origW;
            targetW = maxWidth;
            targetH = Math.round(origH * scale);
        }
        
        this.captureCanvas.width = targetW;
        this.captureCanvas.height = targetH;
        const ctx = this.captureCanvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0, targetW, targetH);
        
        return {
            base64: this.captureCanvas.toDataURL('image/jpeg', quality),
            scale: scale,
            origW: origW,
            origH: origH,
            targetW: targetW,
            targetH: targetH
        };
    }

    /**
     * Captures a base64 encoded JPEG frame from the video stream using the dedicated capture canvas.
     */
    getFrameBase64(quality = 0.8) {
        if (!this.isActive || !this.video.videoWidth) return null;
        
        this.captureCanvas.width = this.video.videoWidth;
        this.captureCanvas.height = this.video.videoHeight;
        const ctx = this.captureCanvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0, this.captureCanvas.width, this.captureCanvas.height);
        
        return this.captureCanvas.toDataURL('image/jpeg', quality);
    }
}
