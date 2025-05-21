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

def main():
    st.title("Maze Solver & Arduino Code Generator")
    st.markdown("Upload a maze image to generate navigation instructions for an Arduino robot.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a maze image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
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
            
            # Check if this is a solved maze with yellow path
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([35, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            yellow_detected = np.sum(yellow_mask) > 1000
            
            # Find path
            path_finder = PathFinder()
            
            if yellow_detected:
                # For solved mazes, extract the path from the yellow line
                st.info("Detected existing solution path in the image. Extracting path...")
                # Find all yellow pixels
                yellow_points = np.where(yellow_mask > 0)
                # Convert to list of (x,y) points
                yellow_coords = list(zip(yellow_points[1], yellow_points[0]))  # Note: x,y are flipped in numpy
                
                # Sample points along the path (to reduce number of points)
                if len(yellow_coords) > 100:
                    step = len(yellow_coords) // 100
                    sampled_path = yellow_coords[::step]
                else:
                    sampled_path = yellow_coords
                
                # Make sure start and end points are included
                if start_point and end_point:
                    # Add start point at beginning
                    sampled_path = [start_point] + sampled_path
                    # Add end point at end
                    sampled_path.append(end_point)
                
                # Use the yellow path directly
                path = sampled_path
                grid_path = path  # Not used for yellow paths
            else:
                # For unsolved mazes, use A* pathfinding
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

if __name__ == "__main__":
    main()
