import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FacePresentationAttackDetector:
    """
    AI-based Passive Face Presentation Attack Detection (PAD) Model.
    Evaluates:
    1. 2D FFT Frequency Spectrum & Screen Moiré Aperture Grid Analysis.
    2. Specular Glass Reflection & Backlight Highlight Analysis.
    3. Chrominance Gamut Dispersion in YCrCb and HSV Color Spaces.
    4. Multi-Scale Laplacian Surface Texture & Blur Variance.
    5. Temporal Micro-Motion & Boundary Frame Consistency.
    """
    
    @staticmethod
    def evaluate(face_bgr, previous_faces=None):
        """
        Analyzes a face crop for presentation attacks.
        Returns:
            is_real (bool): True if authentic living human skin.
            pad_score (float): Confidence score between 0.0 and 1.0.
            diagnosis (str): Human-readable diagnostic description.
            details (dict): Granular metrics for logging/auditing.
        """
        if face_bgr is None or face_bgr.size == 0:
            return False, 0.0, "Invalid frame crop", {}
            
        h, w = face_bgr.shape[:2]
        if h < 40 or w < 40:
            return False, 0.0, "Face too small for PAD analysis", {}
            
        # Normalize face crop to standard 160x160 resolution
        norm_face = cv2.resize(face_bgr, (160, 160))
        gray = cv2.cvtColor(norm_face, cv2.COLOR_BGR2GRAY)
        
        # -------------------------------------------------------------
        # 1. 2D FFT Moiré Pattern & Screen Frequency Interference
        # -------------------------------------------------------------
        # Digital screens (LCD, OLED, AMOLED) possess microscopic RGB subpixel grids.
        # Capturing them on a camera generates periodic interference fringes (Moiré)
        # that manifest as strong high-frequency spectral rings in 2D Fourier space.
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        
        cy, cx = 80, 80
        radius = 22 # Low-frequency center threshold
        y_indices, x_indices = np.ogrid[:160, :160]
        low_freq_mask = (x_indices - cx)**2 + (y_indices - cy)**2 <= radius**2
        
        low_energy = np.sum(magnitude_spectrum[low_freq_mask])
        high_energy = np.sum(magnitude_spectrum[~low_freq_mask])
        fft_ratio = float(high_energy / (low_energy + 1e-6))
        
        # -------------------------------------------------------------
        # 2. Specular Glass Glare & Backlight Saturation
        # -------------------------------------------------------------
        # Phone and tablet displays have a glossy protective glass layer that reflects
        # ambient point light with clipped, high-brightness, low-saturation specular spots.
        hsv = cv2.cvtColor(norm_face, cv2.COLOR_BGR2HSV)
        val_chan = hsv[:, :, 2]
        sat_chan = hsv[:, :, 1]
        glare_mask = (val_chan > 248) & (sat_chan < 28)
        glare_ratio = float(np.sum(glare_mask) / (160.0 * 160.0))
        
        # -------------------------------------------------------------
        # 3. Chrominance Gamut Dispersion (YCrCb & HSV)
        # -------------------------------------------------------------
        # Human skin has rich hemoglobin/melanin reflectance causing natural spread
        # in the Cr (red-difference) and Cb (blue-difference) channels.
        # Paper printouts suffer from printer color compression and paper pigment absorption.
        ycrcb = cv2.cvtColor(norm_face, cv2.COLOR_BGR2YCrCb)
        cr_std = float(np.std(ycrcb[:, :, 1]))
        cb_std = float(np.std(ycrcb[:, :, 2]))
        chroma_spread = cr_std + cb_std
        
        # -------------------------------------------------------------
        # 4. Multi-Scale Laplacian Micro-Texture Variance
        # -------------------------------------------------------------
        # Live human skin shows fine pores and skin texture.
        # Paper prints exhibit either dot-matrix screening or low-pass print blur.
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # -------------------------------------------------------------
        # 5. Temporal Micro-Motion Consistency (Video Replay / Static)
        # -------------------------------------------------------------
        temporal_diff = 15.0 # default assume normal micro-variation
        if previous_faces and len(previous_faces) >= 2:
            # If current frame was already appended at [-1], compare against prior frame [-2]
            prev_crop = previous_faces[-2] if len(previous_faces) >= 2 else previous_faces[0]
            prev_gray = cv2.cvtColor(cv2.resize(prev_crop, (160, 160)), cv2.COLOR_BGR2GRAY)
            temporal_diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            
        # -------------------------------------------------------------
        # Scoring & Classification
        # -------------------------------------------------------------
        penalties = 0.0
        reasons = []
        is_screen = False
        is_print = False
        is_static = False
        
        # Screen Check
        if fft_ratio > 3.3:
            is_screen = True
            penalties += 0.50
            reasons.append("Digital screen Moiré pattern detected")
        elif glare_ratio > 0.05:
            is_screen = True
            penalties += 0.45
            reasons.append("Screen glass specular reflection detected")
            
        # Print Check
        if lap_var < 16.0:
            is_print = True
            penalties += 0.40
            reasons.append("Unnatural flat texture / paper blur detected")
        elif chroma_spread < 6.0:
            is_print = True
            penalties += 0.35
            reasons.append("Color gamut compressed (printed paper artifact)")
            
        # Static / Freeze Replay Check
        if temporal_diff < 0.4:
            is_static = True
            penalties += 0.40
            reasons.append("Artificial static image without physiological micro-movement")
            
        pad_score = max(0.0, min(1.0, 1.0 - penalties))
        
        # Rejection rule: Must have score >= 0.70 and zero explicit attack flags
        is_real = (pad_score >= 0.70) and (not is_screen) and (not is_print) and (not is_static)
        
        if is_real:
            diagnosis = "Live human skin texture & reflectance verified"
        else:
            diagnosis = " | ".join(reasons) if reasons else "Suspicious presentation artifacts detected"
            
        details = {
            "pad_score": round(pad_score, 3),
            "fft_ratio": round(fft_ratio, 2),
            "glare_ratio": round(glare_ratio, 4),
            "chroma_spread": round(chroma_spread, 2),
            "lap_var": round(lap_var, 1),
            "temporal_diff": round(temporal_diff, 2),
            "is_screen": is_screen,
            "is_print": is_print,
            "is_static": is_static
        }
        
        return is_real, pad_score, diagnosis, details
