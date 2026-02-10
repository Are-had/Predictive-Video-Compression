import cv2
import os
import time
import csv

# Import our custom modules
from src.motion import get_motion_vectors
from src.utils import draw_motion_vectors , calculate_psnr
from src.compression import reconstruct_frame, get_residual

# --- CONFIGURATION ---
VIDEO_PATH = 'data/test_video.mp4'         # Input video file
OUTPUT_CSV = 'output/metrics.csv'          # File to save numerical results
OUTPUT_VIDEO = 'output/comparison.mp4'     # File to save the visual result
BLOCK_SIZE = 16                            # Macroblock size (16x16)
SEARCH_AREA = 7                            # Search radius
RESIZE_WIDTH = 320                         # Downscale width for performance

def main():
    # 1. Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    # Initialize Video Capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Get original video properties (FPS is important for the output video)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0: 
        original_fps = 25.0 # Default fallback if FPS detection fails

    prev_gray = None
    frame_count = 0
    
    # --- VIDEO RECORDER SETUP ---
    # We will record a "Side by Side" view (Original Left | Reconstructed Right)
    # The width will be RESIZE_WIDTH * 2. Height is calculated later.
    video_writer = None 

    # --- CSV LOGGER SETUP ---
    # Open the CSV file to write data row by row
    csv_file = open(OUTPUT_CSV, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write the header row
    csv_writer.writerow(['Frame', 'PSNR_dB', 'Time_ms', 'Block_Size'])

    print(f"--- Recording Started ---")
    print(f"Data file: {OUTPUT_CSV}")
    print(f"Video output: {OUTPUT_VIDEO}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Pre-processing: Resize and Convert to Grayscale
        height, width = frame.shape[:2]
        scale = RESIZE_WIDTH / width
        new_height = int(height * scale)
        
        frame_resized = cv2.resize(frame, (RESIZE_WIDTH, new_height))
        current_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Initialize the Video Writer once we know the exact dimensions
        if video_writer is None:
            # MP4V codec for .mp4 files
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Note: We use isColor=False because we are saving grayscale images
            video_writer = cv2.VideoWriter(
                OUTPUT_VIDEO, 
                fourcc, 
                original_fps, 
                (RESIZE_WIDTH * 2, new_height), 
                isColor=False
            )

        # Handle the first frame (we need a previous frame to calculate motion)
        if prev_gray is None:
            prev_gray = current_gray
            continue

        # 3. Core Processing (Motion Estimation + Compensation)
        start_t = time.time()
        
        # A. Find Motion Vectors (The "Encoder" part)
        vectors = get_motion_vectors(prev_gray, current_gray, BLOCK_SIZE, SEARCH_AREA)
        
        # B. Reconstruct Frame using vectors (The "Decoder" part)
        predicted_frame = reconstruct_frame(prev_gray, vectors, BLOCK_SIZE)
        
        dt = (time.time() - start_t) * 1000 # Time in milliseconds

        # 4. Calculate Quality Metric (PSNR)
        psnr = calculate_psnr(current_gray, predicted_frame)

        # 5. Save Data to CSV
        # Format: Frame Number, PSNR (2 decimals), Time, Block Size
        csv_writer.writerow([frame_count, f"{psnr:.2f}", f"{dt:.1f}", BLOCK_SIZE])

        # 6. Save Video Result
        # Create a side-by-side comparison
        combined_view = cv2.hconcat([current_gray, predicted_frame])
        
        # Write to the video file
        video_writer.write(combined_view)

        # 7. Live Display (Optional, just to see progress)
        # We also calculate the Residual for visualization
        residual = get_residual(current_gray, predicted_frame)
        cv2.imshow('Residual Error (Live)', residual * 2) 
        cv2.imshow('Recording Progress...', combined_view)
        
        print(f"Frame {frame_count} | PSNR: {psnr:.2f} dB | Saved.")

        # Prepare for next iteration
        prev_gray = current_gray
        frame_count += 1

        # Quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources
    cap.release()
    if video_writer:
        video_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()
    
    print("\n--- Finished! ---")
    print(f"Results saved in the 'output/' folder.")

if __name__ == "__main__":
    main()