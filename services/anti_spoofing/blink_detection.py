import logging
from services.anti_spoofing.liveness_verifier import SingleBlinkLivenessSession

logger = logging.getLogger(__name__)

class BlinkDetector:
    """
    Backwards-compatible wrapper that aliases to SingleBlinkLivenessSession.
    """
    def __init__(self, client_id="default"):
        self.session = SingleBlinkLivenessSession(client_id)
        
    def reset(self):
        self.session.reset()
        
    def check_liveness(self, frame_bgr, detection):
        is_completed, status, details = self.session.process_frame(frame_bgr, detection)
        return is_completed, status, details
