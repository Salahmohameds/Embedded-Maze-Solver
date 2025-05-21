import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image

from maze_processor import MazeProcessor
from path_finder import PathFinder
from code_generator import ArduinoCodeGenerator, EmbeddedCCodeGenerator

st.set_page_config(
    page_title="Maze Solver & Arduino Code Generator",
    page_icon="🤖",
    layout="wide"
)

def extract_yellow_path(image):
    """
    Extract the yellow path from the maze image
    
    Args:
        image: Input image as numpy array
        
    Returns:
        path_img: Image with only the yellow path
        path_points: List of (x, y) coordinates along the path
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define yellow color range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    
    # Create mask for yellow path
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Extract path points
    path_points = cv2.findNonZero(mask)
    
    # Create path image
    path_img = np.zeros_like(image)
    if path_points is not None:
        for point in path_points:
            x, y = point[0]
            cv2.circle(path_img, (x, y), 2, (0, 255, 0), -1)
    
    # Extract only the yellow path from original image
    yellow_path = cv2.bitwise_and(image, image, mask=mask)
    
    return yellow_path, path_points

def main():
    st.title("Maze Solver & Arduino Code Generator")
    st.markdown("Upload a maze image to generate navigation instructions for an Arduino robot.")
    
    # Add tabs for different modes
    tab1, tab2 = st.tabs(["Solve Maze", "Extract Yellow Path"])
    
    # File uploader (shared between tabs)
    uploaded_file = st.file_uploader("Choose a maze image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Tab 1: Solve Maze
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Maze Image")
                st.image(img_array, caption="Original Maze", use_container_width=True)
            
            # Process the image
            with st.spinner("Processing maze image..."):
                maze_processor = MazeProcessor()
                
                # Process image and detect walls/paths
                processed_img, binary_maze, start_point, end_point = maze_processor.process_image(img_array)
                
                if start_point is None or end_point is None:
                    st.warning("Could not detect start and end points clearly. Using default positions.")
                    # Use fallback positions (top-right and bottom-left)
                    height, width = binary_maze.shape
                    start_point = (width - 30, 30)
                    end_point = (30, height - 30)
                
                # For all mazes, use A* pathfinding to solve the maze
                st.info("Solving maze using A* pathfinding algorithm...")
                
                # Find path
                path_finder = PathFinder()
                path, grid_path = path_finder.find_path(binary_maze, start_point, end_point)
                
                if path is None or len(path) == 0:
                    st.error("Could not find a valid path through the maze.")
                    return
                
                # Draw the path on the image
                path_img = maze_processor.draw_path(img_array.copy(), path)
                
                with col2:
                    st.subheader("Processed Maze with Path")
                    st.image(path_img, caption="Path Solution", use_container_width=True)
                
                # Generate movement commands
                arduino_generator = ArduinoCodeGenerator()
                movement_commands = arduino_generator.generate_movement_commands(path, grid_path)
                
                # Generate Arduino code
                arduino_code = arduino_generator.generate_complete_sketch(movement_commands)
                
                # Generate Embedded C code
                embedded_c_generator = EmbeddedCCodeGenerator()
                embedded_c_code = embedded_c_generator.generate_complete_code(movement_commands)
                
                # Display code
                st.subheader("Generated Arduino Code")
                st.code(arduino_code, language="cpp")
                
                # Button to copy code
                st.download_button(
                    label="Download Arduino Code",
                    data=arduino_code,
                    file_name="maze_solver.ino",
                    mime="text/plain"
                )
                
                st.subheader("Generated Embedded C Code")
                st.code(embedded_c_code, language="c")
                
                # Button to copy code
                st.download_button(
                    label="Download Embedded C Code",
                    data=embedded_c_code,
                    file_name="maze_solver.c",
                    mime="text/plain"
                )
                
                # Display validation info
                st.subheader("Path Validation")
                total_distance = sum([cmd.get('duration', 0) for cmd in movement_commands if cmd.get('type') == 'F'])
                total_turns = sum([1 for cmd in movement_commands if cmd.get('type') in ['L', 'R']])
                
                st.info(f"Path Statistics: {len(path)} total steps, {total_distance}mm distance, {total_turns} turns")
        
        # Tab 2: Extract Yellow Path
        with tab2:
            st.subheader("Yellow Path Extraction")
            
            # Process to extract yellow path
            yellow_path, path_points = extract_yellow_path(img_array)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_array, caption="Original Image", use_container_width=True)
            
            with col2:
                st.image(yellow_path, caption="Extracted Yellow Path", use_container_width=True)
            
            # Show path points
            if path_points is not None and len(path_points) > 0:
                # Sample points for display (too many would be overwhelming)
                if len(path_points) > 100:
                    sample_rate = len(path_points) // 100
                    sampled_points = path_points[::sample_rate]
                else:
                    sampled_points = path_points
                
                # Create a new image with the path highlighted
                path_overlay = img_array.copy()
                for point in sampled_points:
                    x, y = point[0]
                    cv2.circle(path_overlay, (x, y), 3, (0, 255, 0), -1)
                
                st.subheader("Path Points Visualization")
                st.image(path_overlay, caption="Green dots show extracted path points", use_container_width=True)
                
                # Convert to Python code
                path_points_list = [(p[0][0], p[0][1]) for p in path_points[:20]]  # Show first 20 for example
                
                st.subheader("Python Code for Path Extraction")
                python_code = f"""
import cv2
import numpy as np

# Load the maze image
image = cv2.imread("solved-maze.jpg")

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define yellow color range
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

# Create mask for yellow path
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Clean up mask
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Extract path points
path_points = cv2.findNonZero(mask)

# Sample of extracted points:
# {path_points_list}

# Visualize path
path_overlay = image.copy()
for point in path_points:
    x, y = point[0]
    cv2.circle(path_overlay, (x, y), 3, (0, 255, 0), -1)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Yellow Path Mask", mask)
cv2.imshow("Path Visualization", path_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
                st.code(python_code, language="python")
                
                # Provide download button for the code
                st.download_button(
                    label="Download Path Extraction Code",
                    data=python_code,
                    file_name="extract_yellow_path.py",
                    mime="text/plain"
                )
                
                # Display total number of points
                st.success(f"Successfully extracted {len(path_points)} points from the yellow path!")
            else:
                st.error("No yellow path detected in the image.")

if __name__ == "__main__":
    main()
