import numpy as np

def calculate_ear(eye_points):
    """
    Calculates the Eye Aspect Ratio (EAR) given 6 landmark points.
    Formula: EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    if len(eye_points) < 6:
        return 0.0
        
    # Convert points to numpy arrays
    p1 = np.array(eye_points[0])
    p2 = np.array(eye_points[1])
    p3 = np.array(eye_points[2])
    p4 = np.array(eye_points[3])
    p5 = np.array(eye_points[4])
    p6 = np.array(eye_points[5])
    
    # Compute Euclidean distances between vertical coordinates
    dist_v1 = np.linalg.norm(p2 - p6)
    dist_v2 = np.linalg.norm(p3 - p5)
    
    # Compute Euclidean distance between horizontal coordinates
    dist_h = np.linalg.norm(p1 - p4)
    
    if dist_h == 0:
        return 0.0
        
    # EAR Formula
    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return ear
