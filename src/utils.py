import cv2
import numpy as np

def draw_motion_vectors(frame, vectors, color=(0, 255, 0)):
    """
    Draws motion vectors (arrows) on the frame.
    
    Args:
        frame: The image on which to draw.
        vectors: List of dictionaries {x, y, u, v, sad}.
        color: Color of the arrows (B, G, R). Default is Green.
        
    Returns:
        The frame with arrows drawn on it.
    """
    output_frame = frame.copy()
    
    for v in vectors:
        p1 = (v['x'], v['y'])
        p2 = (v['x'] + v['u'], v['y'] + v['v'])

        if abs(v['u']) < 1 and abs(v['v']) < 1:
            continue
            
        cv2.arrowedLine(output_frame, p1, p2, color, 1, tipLength=0.3)


    return output_frame