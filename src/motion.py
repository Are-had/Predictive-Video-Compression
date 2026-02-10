import numpy as np

def sad(block1, block2):
    """
    Calculates the Sum of Absolute Differences (SAD) between two blocks.
    Lower result means the blocks are more similar.
    """

    return np.sum(np.abs(np.subtract(block1, block2, dtype=np.float32)))

def get_motion_vectors(reference_frame, current_frame, block_size=16, search_area=16):
    """
    Estimates motion between two frames using Full Search Block Matching.
    
    Args:
        reference_frame: The previous frame (N-1) in grayscale.
        current_frame: The current frame (N) in grayscale.
        block_size: Size of the macroblock (e.g., 16x16 pixels).
        search_area: The search radius (e.g., look +/- 16 pixels around).
        
    Returns:
        vectors: A list of dictionaries containing {x, y, u, v, sad} for each block.
                 (x,y) = block position, (u,v) = motion vector.
    """
    
    # Get dimensions of the current frame
    height, width = current_frame.shape
    vectors = []

    # 1. Loop through the current frame block by block
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            
            # Extract the target block from the current frame
            target_block = current_frame[y:y+block_size, x:x+block_size]
            
            # Initialize best match variables
            best_sad = float('inf') 
            best_vector = (0, 0)   

            # 2. Define the search window in the reference frame (Previous Frame)
            y_start = max(0, y - search_area)
            y_end = min(height - block_size, y + search_area)
            x_start = max(0, x - search_area)
            x_end = min(width - block_size, x + search_area)

            # 3. The Search Loop (Full Search)
            for r_y in range(y_start, y_end):
                for r_x in range(x_start, x_end):
                    
                    # Extract the candidate block from the reference frame
                    candidate_block = reference_frame[r_y:r_y+block_size, r_x:r_x+block_size]

                    if candidate_block.shape != target_block.shape:
                        continue

                    # CALCULATE COST (SAD)
                    cost = sad(target_block, candidate_block)

                    if cost < best_sad:
                        best_sad = cost
                        best_vector = (r_x - x, r_y - y)

            # 4. Store the result
            vectors.append({
                'x': x, 
                'y': y, 
                'u': best_vector[0], 
                'v': best_vector[1],
                'sad': best_sad
            })
            
    return vectors