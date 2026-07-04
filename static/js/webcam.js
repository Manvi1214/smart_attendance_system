/**
 * Shared webcam utility for capturing frames in browser canvas.
 */
class WebcamHelper {
    constructor(videoElement, canvasElement = null) {
        this.video = videoElement;
        this.canvas = canvasElement || document.createElement('canvas');
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
     * Captures a base64 encoded JPEG frame from the video stream.
     */
    getFrameBase64(quality = 0.8) {
        if (!this.isActive || !this.video.videoWidth) return null;
        
        const ctx = this.canvas.getContext('2d');
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        // Draw video frame to hidden canvas
        ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Output base64 data url
        return this.canvas.toDataURL('image/jpeg', quality);
    }
}
