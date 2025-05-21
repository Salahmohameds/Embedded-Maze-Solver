import streamlit as st
import cv2
import numpy as np
from PIL import Image

from maze_processor import MazeProcessor
from path_finder import PathFinder
from code_generator import ArduinoCodeGenerator

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
        yellow_path: Image with only the yellow path
        path_points: List of (x, y) coordinates along the path
        ordered_points: Ordered list of points from start to end
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
    
    # Extract only the yellow path from original image
    yellow_path = cv2.bitwise_and(image, image, mask=mask)
    
    # Try to order the points from start to end
    ordered_points = []
    if path_points is not None and len(path_points) > 0:
        # Find start and end points (red and green areas)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Red range (for START)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Green range (for END)
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        
        # Create masks for start/end
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find center of start/end areas
        start_point = None
        end_point = None
        
        # Find the closest yellow point to start/end
        if np.sum(mask_red) > 0:
            red_points = np.where(mask_red > 0)
            start_y = int(np.mean(red_points[0]))
            start_x = int(np.mean(red_points[1]))
            start_point = (start_x, start_y)
        
        if np.sum(mask_green) > 0:
            green_points = np.where(mask_green > 0)
            end_y = int(np.mean(green_points[0]))
            end_x = int(np.mean(green_points[1]))
            end_point = (end_x, end_y)
        
        # If couldn't find colors, use image corners
        if start_point is None:
            start_point = (image.shape[1]-50, 50)  # Top-right
        
        if end_point is None:
            end_point = (50, image.shape[0]-50)    # Bottom-left
        
        # Find points in the yellow path closest to start and end
        min_start_dist = float('inf')
        min_end_dist = float('inf')
        start_index = 0
        end_index = 0
        
        for i, point in enumerate(path_points):
            x, y = point[0]
            
            # Distance to start
            start_dist = np.sqrt((x - start_point[0])**2 + (y - start_point[1])**2)
            if start_dist < min_start_dist:
                min_start_dist = start_dist
                start_index = i
            
            # Distance to end
            end_dist = np.sqrt((x - end_point[0])**2 + (y - end_point[1])**2)
            if end_dist < min_end_dist:
                min_end_dist = end_dist
                end_index = i
        
        # Sample points along path for simplified representation
        # Note: This is a simplistic approach. A more sophisticated algorithm
        # would trace the path from start to end more accurately.
        if len(path_points) > 50:
            # Determine how many points to sample
            sample_rate = max(1, len(path_points) // 50)
            
            # Extract points, starting near the start point
            point_array = np.array([p[0] for p in path_points])
            
            # Get a simplified path by sampling
            ordered_points = point_array[::sample_rate].tolist()
            
            # Ensure start and end points are included
            if start_index < end_index:
                ordered_points = [path_points[start_index][0].tolist()] + ordered_points + [path_points[end_index][0].tolist()]
            else:
                ordered_points = [path_points[end_index][0].tolist()] + ordered_points + [path_points[start_index][0].tolist()]
        else:
            # If few points, use all of them
            ordered_points = [p[0].tolist() for p in path_points]
    
    return yellow_path, path_points, ordered_points

def generate_arduino_code_from_points(path_points):
    """Generate Arduino code from extracted path points"""
    
    # Convert path points to structured path format
    structured_path = []
    for i, point in enumerate(path_points):
        structured_path.append((point[0], point[1]))
    
    # Create artificial grid_path for compatibility
    grid_path = [tuple(p) for p in structured_path]
    
    # Use existing ArduinoCodeGenerator to generate code
    arduino_generator = ArduinoCodeGenerator()
    movement_commands = arduino_generator.generate_movement_commands(structured_path, grid_path)
    arduino_code = arduino_generator.generate_complete_sketch(movement_commands)
    
    # Return only Arduino code and movement commands
    return arduino_code, movement_commands

def main():
    st.title("Maze Solver & Arduino Code Generator")
    st.markdown("Upload a maze image to generate Arduino code for your robot. The app can process both regular maze images and mazes with yellow solution paths.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a maze image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Display original image
        st.subheader("Original Maze Image")
        st.image(img_array, caption="Original Maze", use_container_width=True)
        
        # Check if image has a yellow path
        yellow_path, path_points, ordered_points = extract_yellow_path(img_array)
        has_yellow_path = path_points is not None and len(path_points) > 0
        
        # Create tabs based on image content
        if has_yellow_path:
            option = st.radio(
                "Select processing method:",
                ["Use yellow path from image", "Solve maze from scratch"]
            )
            
            if option == "Use yellow path from image":
                # YELLOW PATH PROCESSING
                with st.spinner("Extracting yellow path and generating code..."):
                    # Show extracted path
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(yellow_path, caption="Extracted Yellow Path", use_container_width=True)
                    
                    # Create visualization with ordered points
                    path_overlay = img_array.copy()
                    for x, y in ordered_points:
                        cv2.circle(path_overlay, (int(x), int(y)), 3, (0, 255, 0), -1)
                    
                    with col2:
                        st.image(path_overlay, caption="Path Points", use_container_width=True)
                    
                    # Generate Arduino code directly from extracted path
                    arduino_code, movement_commands = generate_arduino_code_from_points(ordered_points)
                    
                    # Display Arduino code (main focus)
                    st.subheader("🤖 Arduino Code for Your Robot")
                    st.code(arduino_code, language="cpp")
                    
                    # Button to download
                    st.download_button(
                        label="Download Arduino Code (.ino)",
                        data=arduino_code,
                        file_name="maze_solver.ino",
                        mime="text/plain"
                    )
                    
                    # Display path statistics
                    total_distance = sum([cmd.get('duration', 0) for cmd in movement_commands if cmd.get('type') == 'F'])
                    total_turns = sum([1 for cmd in movement_commands if cmd.get('type') in ['L', 'R']])
                    
                    st.success(f"✅ Successfully generated Arduino code with {len(ordered_points)} path points, {total_distance}mm travel distance, and {total_turns} turns")
                
            else:
                # STANDARD MAZE SOLVING
                with st.spinner("Processing maze and finding optimal path..."):
                    # Process image to detect maze structure
                    maze_processor = MazeProcessor()
                    processed_img, binary_maze, start_point, end_point = maze_processor.process_image(img_array)
                    
                    if start_point is None or end_point is None:
                        st.warning("Could not detect start and end points clearly. Using default positions.")
                        height, width = binary_maze.shape
                        start_point = (width - 30, 30)
                        end_point = (30, height - 30)
                    
                    # Solve the maze using A*
                    path_finder = PathFinder()
                    path, grid_path = path_finder.find_path(binary_maze, start_point, end_point)
                    
                    if path is None or len(path) == 0:
                        st.error("Could not find a valid path through the maze. Try again with a clearer image.")
                        return
                    
                    # Draw path on image
                    path_img = maze_processor.draw_path(img_array.copy(), path)
                    st.image(path_img, caption="Computed Path Solution", use_container_width=True)
                    
                    # Generate Arduino code
                    arduino_generator = ArduinoCodeGenerator()
                    movement_commands = arduino_generator.generate_movement_commands(path, grid_path)
                    arduino_code = arduino_generator.generate_complete_sketch(movement_commands)
                    
                    # Display Arduino code (main focus)
                    st.subheader("🤖 Arduino Code for Your Robot")
                    st.code(arduino_code, language="cpp")
                    
                    # Button to download
                    st.download_button(
                        label="Download Arduino Code (.ino)",
                        data=arduino_code,
                        file_name="maze_solver.ino",
                        mime="text/plain"
                    )
                    
                    # Display path statistics
                    total_distance = sum([cmd.get('duration', 0) for cmd in movement_commands if cmd.get('type') == 'F'])
                    total_turns = sum([1 for cmd in movement_commands if cmd.get('type') in ['L', 'R']])
                    
                    st.success(f"✅ Successfully generated Arduino code with {len(path)} path points, {total_distance}mm travel distance, and {total_turns} turns")
        
        else:
            # STANDARD MAZE SOLVING (No yellow path detected)
            with st.spinner("Processing maze and finding optimal path..."):
                # Process image to detect maze structure
                maze_processor = MazeProcessor()
                processed_img, binary_maze, start_point, end_point = maze_processor.process_image(img_array)
                
                if start_point is None or end_point is None:
                    st.warning("Could not detect start and end points clearly. Using default positions.")
                    height, width = binary_maze.shape
                    start_point = (width - 30, 30)
                    end_point = (30, height - 30)
                
                # Solve the maze using A*
                path_finder = PathFinder()
                path, grid_path = path_finder.find_path(binary_maze, start_point, end_point)
                
                if path is None or len(path) == 0:
                    st.error("Could not find a valid path through the maze. Try again with a clearer image.")
                    return
                
                # Draw path on image
                path_img = maze_processor.draw_path(img_array.copy(), path)
                st.image(path_img, caption="Computed Path Solution", use_container_width=True)
                
                # Generate Arduino code
                arduino_generator = ArduinoCodeGenerator()
                movement_commands = arduino_generator.generate_movement_commands(path, grid_path)
                arduino_code = arduino_generator.generate_complete_sketch(movement_commands)
                
                # Display Arduino code (main focus)
                st.subheader("🤖 Arduino Code for Your Robot")
                st.code(arduino_code, language="cpp")
                
                # Button to download
                st.download_button(
                    label="Download Arduino Code (.ino)",
                    data=arduino_code,
                    file_name="maze_solver.ino",
                    mime="text/plain"
                )
                
                # Display path statistics
                total_distance = sum([cmd.get('duration', 0) for cmd in movement_commands if cmd.get('type') == 'F'])
                total_turns = sum([1 for cmd in movement_commands if cmd.get('type') in ['L', 'R']])
                
                st.success(f"✅ Successfully generated Arduino code with {len(path)} path points, {total_distance}mm travel distance, and {total_turns} turns")

if __name__ == "__main__":
    main()
