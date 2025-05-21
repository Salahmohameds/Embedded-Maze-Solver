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
        
        # Detect start and end points
        start_point, end_point = self._detect_start_end_points(image)
        
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
        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Try to detect by color markers (red for start, green for end)
        start_mask = cv2.inRange(hsv, self.start_color_lower, self.start_color_upper)
        end_mask = cv2.inRange(hsv, self.end_color_lower, self.end_color_upper)
        
        # Find contours for start and end
        start_contours, _ = cv2.findContours(start_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        end_contours, _ = cv2.findContours(end_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        start_point = None
        end_point = None
        
        # Get start point from contours
        if start_contours:
            largest_start = max(start_contours, key=cv2.contourArea)
            start_moment = cv2.moments(largest_start)
            if start_moment["m00"] != 0:
                start_x = int(start_moment["m10"] / start_moment["m00"])
                start_y = int(start_moment["m01"] / start_moment["m00"])
                start_point = (start_x, start_y)
        
        # Get end point from contours
        if end_contours:
            largest_end = max(end_contours, key=cv2.contourArea)
            end_moment = cv2.moments(largest_end)
            if end_moment["m00"] != 0:
                end_x = int(end_moment["m10"] / end_moment["m00"])
                end_y = int(end_moment["m01"] / end_moment["m00"])
                end_point = (end_x, end_y)
        
        # If color detection fails, try to determine by position
        if start_point is None or end_point is None:
            # Check the top right corner for "Start" text
            top_right = (image.shape[1] - 50, 50)
            bottom_left = (50, image.shape[0] - 50)
            
            # Assuming start is top-right and end is bottom-left
            # (common in many maze images)
            if start_point is None:
                start_point = top_right
            if end_point is None:
                end_point = bottom_left
        
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
