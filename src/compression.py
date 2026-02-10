import numpy as np



def reconstruct_frame(reference_frame, vectors, block_size=16):
    """
    Reconstructs the current frame (Prediction) using the reference frame and motion vectors.
    This simulates the 'Decoder' part of a codec.
    """
    height, width = reference_frame.shape
    reconstructed = np.zeros((height, width), dtype=np.uint8)
    
    for v in vectors:
        # Current block *
        x, y = v['x'], v['y']
        
        # Motion Vector
        u, v_y = v['u'], v['v']

        ref_x = x + u
        ref_y = y + v_y
        
        # Clip coordinates to stay inside image bounds
        ref_x = int(max(0, min(width - block_size, ref_x)))
        ref_y = int(max(0, min(height - block_size, ref_y)))
        
        # Copy the block from reference to reconstructed frame
        block = reference_frame[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
        
        # Handle edge cases where block size might differ near borders
        h_block, w_block = block.shape
        reconstructed[y:y+h_block, x:x+w_block] = block
        
    return reconstructed

def get_residual(current_frame, predicted_frame):
    """
    Calculates the Residual (The difference between Reality and Prediction).
    In a real codec, this is the only image data transmitted.
    """
    # Use int16 to handle negative results, then take absolute value
    diff = np.abs(current_frame.astype(np.int16) - predicted_frame.astype(np.int16))
    
    # Clip back to 0-255 uint8 range
    return np.clip(diff, 0, 255).astype(np.uint8)