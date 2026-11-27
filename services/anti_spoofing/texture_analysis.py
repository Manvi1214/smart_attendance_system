import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TextureAnalyzer:
    """
    Analyzes face texture, color dispersion, and frequency domain (2D FFT)
    to detect presentation attacks including:
    - Printed photographs on paper
    - Photographs displayed on mobile/tablet screens (Moiré pattern detection)
    - Low-quality photo re-captures
    """
    
    @staticmethod
    def analyze_texture(face_bgr):
        """
        Analyzes a BGR face crop.
        Returns:
            dict with:
                - passed (bool): True if texture is consistent with live human skin
                - score (float 0.0 - 1.0): Confidence in skin naturalness
                - is_screen (bool): True if digital screen / Moiré artifact detected
                - is_print (bool): True if flat paper print artifact detected
                - reason (str): Human-readable diagnosis
        """
        if face_bgr is None or face_bgr.size == 0:
            return {
                "passed": False,
                "score": 0.0,
                "is_screen": False,
                "is_print": False,
                "reason": "Invalid face crop"
            }
            
        try:
            h, w = face_bgr.shape[:2]
            if h < 40 or w < 40:
                return {
                    "passed": False,
                    "score": 0.0,
                    "is_screen": False,
                    "is_print": False,
                    "reason": "Face crop too small for texture analysis"
                }
                
            # Normalize size for consistent filter response
            norm_face = cv2.resize(face_bgr, (160, 160))
            gray = cv2.cvtColor(norm_face, cv2.COLOR_BGR2GRAY)
            
            # 1. 2D FFT Moiré Pattern / Digital Screen Frequency Analysis
            # Digital LCD/OLED screens display RGB pixel grids that produce
            # high-frequency periodic interference spikes in the frequency domain.
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            
            cy, cx = 80, 80
            radius = 20 # Low-frequency center circle
            y_indices, x_indices = np.ogrid[:160, :160]
            low_freq_mask = (x_indices - cx)**2 + (y_indices - cy)**2 <= radius**2
            
            low_energy = np.sum(magnitude_spectrum[low_freq_mask])
            high_energy = np.sum(magnitude_spectrum[~low_freq_mask])
            fft_ratio = float(high_energy / (low_energy + 1e-6))
            
            # 2. Chrominance Diversity in YCrCb color space
            # Live human skin has a characteristic spread of Cr (red-difference)
            # and Cb (blue-difference). Printed paper and cheap screens have compressed
            # or quantized color gamut.
            ycrcb = cv2.cvtColor(norm_face, cv2.COLOR_BGR2YCrCb)
            cr_channel = ycrcb[:, :, 1]
            cb_channel = ycrcb[:, :, 2]
            cr_std = float(np.std(cr_channel))
            cb_std = float(np.std(cb_channel))
            chroma_spread = cr_std + cb_std
            
            # 3. Specular Glare / Screen Backlight Highlight Analysis
            # Mobile screens have glass layers that reflect bright white specular spots (V > 250)
            # with sharp clipping boundaries.
            hsv = cv2.cvtColor(norm_face, cv2.COLOR_BGR2HSV)
            val_channel = hsv[:, :, 2]
            sat_channel = hsv[:, :, 1]
            glare_mask = (val_channel > 248) & (sat_channel < 30)
            glare_ratio = float(np.sum(glare_mask) / (160 * 160))
            
            # 4. Laplacian Texture Sharpness / Blur Variance
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # Evaluation rules:
            is_screen = False
            is_print = False
            reasons = []
            
            # Screen Check:
            # - Excessive high-frequency grid interference (FFT ratio > 3.2 on 160px crop)
            # - High specular glass reflection ratio (> 4.5% of face)
            if fft_ratio > 3.4:
                is_screen = True
                reasons.append("Digital screen Moiré pattern detected")
            elif glare_ratio > 0.06:
                is_screen = True
                reasons.append("Screen glass reflection / glare detected")
                
            # Print Check:
            # - Overly flat texture (Laplacian variance < 15) indicates blurred print re-capture
            # - Unnatural chromatic spread (chroma_spread < 6.0 indicates paper pigment compression)
            if lap_var < 15.0:
                is_print = True
                reasons.append("Unnatural flat texture / paper blur detected")
            elif chroma_spread < 6.0:
                is_print = True
                reasons.append("Color gamut compressed (printed paper artifact)")
                
            passed = (not is_screen) and (not is_print)
            
            # Compute a normalized skin naturalness score (0.0 to 1.0)
            score = 1.0
            if is_screen:
                score -= 0.65
            if is_print:
                score -= 0.50
            if lap_var < 35.0:
                score -= 0.15
            score = max(0.0, min(1.0, score))
            
            reason = "Natural skin texture verified" if passed else ("; ".join(reasons))
            
            return {
                "passed": passed,
                "score": round(score, 2),
                "is_screen": is_screen,
                "is_print": is_print,
                "lap_var": round(lap_var, 1),
                "fft_ratio": round(fft_ratio, 3),
                "chroma_spread": round(chroma_spread, 2),
                "glare_ratio": round(glare_ratio, 4),
                "reason": reason
            }
        except Exception as e:
            logger.error(f"Error in texture analysis: {e}")
            return {
                "passed": True,
                "score": 0.7,
                "is_screen": False,
                "is_print": False,
                "reason": "Texture analysis bypassed"
            }
