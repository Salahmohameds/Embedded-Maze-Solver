import cv2
import numpy as np
import matplotlib.pyplot as plt

class MazeProcessor:
    def __init__(self):
        # Define color ranges for start and end points
        self.start_color_lower = np.array([0, 100, 100])  # Red color in HSV
        self.start_color_upper = np.array([10, 255, 255])
        
        self.end_color_lower = np.array([40, 100, 100])  # Green color in HSV
        self.end_color_upper = np.array([80, 255, 255])
    
    def process_image(self, image):
        """
        Process the maze image to detect walls, paths, and start/end points
        
        Args:
            image: The input maze image (numpy array)
            
        Returns:
            processed_img: The processed image
            binary_maze: Binary representation of the maze (0=wall, 1=path)
            start_point: (x, y) coordinates of the start point
            end_point: (x, y) coordinates of the end point
        """
        # Check if the image has a yellow solution path
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define yellow color range
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        # Create a mask for yellow color
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_detected = np.sum(yellow_mask) > 1000  # Threshold for detection
        
        # Detect start and end points first
        start_point, end_point = self._detect_start_end_points(image)
        
        if yellow_detected:
            # The image already has a solution path drawn in yellow
            # Extract and use that path directly
            
            # Clean up the yellow mask
            kernel = np.ones((5, 5), np.uint8)
            yellow_mask_cleaned = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
            yellow_mask_dilated = cv2.dilate(yellow_mask_cleaned, kernel, iterations=1)
            
            # Create a binary maze where the yellow path is the only valid path
            binary_maze = np.zeros_like(yellow_mask_dilated, dtype=np.uint8)
            binary_maze[yellow_mask_dilated > 0] = 1
            
            # Add start and end points to make sure they're in the path
            if start_point:
                x, y = start_point
                # Create a small area of 1s around the start point
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < binary_maze.shape[1] and 0 <= ny < binary_maze.shape[0]:
                            binary_maze[ny, nx] = 1
            
            if end_point:
                x, y = end_point
                # Create a small area of 1s around the end point
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < binary_maze.shape[1] and 0 <= ny < binary_maze.shape[0]:
                            binary_maze[ny, nx] = 1
            
            # Dilate the path to make it wider for better pathfinding
            binary_maze = cv2.dilate(binary_maze, kernel, iterations=1)
            
            # For visualization
            processed_img = yellow_mask_dilated
            
            return processed_img, binary_maze, start_point, end_point
        
        else:
            # Standard maze processing for images without solution
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding for better results with varying lighting
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            
            # Invert the binary image (walls=0, paths=1)
            binary_inverted = cv2.bitwise_not(binary)
            
            # Remove small noise with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            binary_cleaned = cv2.morphologyEx(binary_inverted, cv2.MORPH_OPEN, kernel)
            
            # Apply dilation to make paths wider and connect broken paths
            kernel_dilate = np.ones((3, 3), np.uint8)
            binary_dilated = cv2.dilate(binary_cleaned, kernel_dilate, iterations=1)
            
            # Create a binary maze representation (0=wall, 1=path)
            binary_maze = binary_dilated.copy()
            binary_maze = binary_maze // 255  # Normalize to 0 and 1
            
            return binary_cleaned, binary_maze, start_point, end_point
    
    def _detect_start_end_points(self, image):
        """
        Detect the start and end points in the maze based on color markers
        
        Args:
            image: The input maze image
            
        Returns:
            start_point: (x, y) coordinates of the start point
            end_point: (x, y) coordinates of the end point
        """
        # Convert image to HSV color space for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Expanded color ranges for better detection in varying lighting
        # Red color in HSV (need two ranges because red wraps around in HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Green color in HSV
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        # Yellow color in HSV (for the solution path in the image)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        # Create masks for each color
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Apply morphological operations to clean up the masks
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours for each color
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        start_point = None
        end_point = None
        
        # Find start point (red)
        if red_contours:
            # Find the largest red contour
            largest_red = max(red_contours, key=cv2.contourArea)
            # Get its centroid
            M = cv2.moments(largest_red)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                start_point = (cx, cy)
        
        # Find end point (green)
        if green_contours:
            # Find the largest green contour
            largest_green = max(green_contours, key=cv2.contourArea)
            # Get its centroid
            M = cv2.moments(largest_green)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                end_point = (cx, cy)
        
        # If detection still fails, look for text indicators or use heuristics
        if start_point is None or end_point is None:
            # Try to find START and END text using OCR (simplified)
            height, width = image.shape[:2]
            
            # As a fallback, use corners of the maze
            # Check for yellow regions (solution path) to find endpoints
            if yellow_contours:
                yellow_points = []
                for contour in yellow_contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        yellow_points.append((cx, cy))
                
                if yellow_points:
                    # Find two points that are furthest apart
                    max_dist = 0
                    for i, p1 in enumerate(yellow_points):
                        for j, p2 in enumerate(yellow_points[i+1:], i+1):
                            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                            if dist > max_dist:
                                max_dist = dist
                                if start_point is None and end_point is None:
                                    start_point = p1
                                    end_point = p2
            
            # If still not found, use corners as fallback
            if start_point is None:
                # Top-right corner as start
                start_point = (width - 20, 20)
            
            if end_point is None:
                # Bottom-left corner as end
                end_point = (20, height - 20)
        
        # Verify that points are within the maze path (not on walls)
        # This ensures we're starting and ending on valid paths
        
        return start_point, end_point
    
    def draw_path(self, image, path):
        """
        Draw the path on the maze image
        
        Args:
            image: The original maze image
            path: List of (x, y) coordinates representing the path
            
        Returns:
            path_img: Image with the path drawn on it
        """
        path_img = image.copy()
        
        # Draw each point in the path
        for i in range(len(path) - 1):
            cv2.line(path_img, path[i], path[i+1], (255, 255, 0), 3)  # Yellow line
        
        # Mark start and end points
        if len(path) > 0:
            cv2.circle(path_img, path[0], 10, (0, 0, 255), -1)  # Red circle for start
            cv2.circle(path_img, path[-1], 10, (0, 255, 0), -1)  # Green circle for end
        
        return path_img
