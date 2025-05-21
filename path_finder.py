import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement

class PathFinder:
    def __init__(self):
        """Initialize the PathFinder with A* algorithm"""
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    
    def find_path(self, binary_maze, start_point, end_point):
        """
        Find the optimal path through the maze using A* algorithm
        
        Args:
            binary_maze: Binary representation of the maze (0=wall, 1=path)
            start_point: (x, y) coordinates of the start point
            end_point: (x, y) coordinates of the end point
            
        Returns:
            path: List of (x, y) coordinates representing the optimal path
            grid_path: List of grid cells in the path (for direction calculation)
        """
        # Create a grid from the binary maze
        # Note: We need to transpose the maze for the Grid class
        height, width = binary_maze.shape
        grid = Grid(width=width, height=height, matrix=binary_maze.tolist())
        
        # Convert start and end points to grid coordinates
        start_x, start_y = start_point
        end_x, end_y = end_point
        
        # Ensure points are within grid bounds
        start_x = max(0, min(start_x, width - 1))
        start_y = max(0, min(start_y, height - 1))
        end_x = max(0, min(end_x, width - 1))
        end_y = max(0, min(end_y, height - 1))
        
        # Create start and end nodes
        start_node = grid.node(start_x, start_y)
        end_node = grid.node(end_x, end_y)
        
        # Find the path
        path, runs = self.finder.find_path(start_node, end_node, grid)
        
        # Convert path to pixel coordinates
        pixel_path = [(node.x, node.y) for node in path]
        
        # Apply path smoothing
        smoothed_path = self._smooth_path(pixel_path, binary_maze)
        
        return smoothed_path, path
    
    def _smooth_path(self, path, binary_maze):
        """
        Smooth the path to remove unnecessary points and make turns more natural
        
        Args:
            path: List of (x, y) coordinates representing the path
            binary_maze: Binary representation of the maze
            
        Returns:
            smoothed_path: Smoothed path with fewer points
        """
        if len(path) <= 2:
            return path
        
        # Initialize smoothed path with start point
        smoothed_path = [path[0]]
        
        # Current direction
        curr_dir = None
        
        for i in range(1, len(path)):
            # Get current and previous points
            prev_x, prev_y = path[i-1]
            curr_x, curr_y = path[i]
            
            # Calculate direction
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            
            # Skip points that maintain the same direction
            new_dir = (dx, dy)
            if curr_dir is None or new_dir != curr_dir:
                curr_dir = new_dir
                smoothed_path.append((curr_x, curr_y))
        
        # Always include the end point
        if smoothed_path[-1] != path[-1]:
            smoothed_path.append(path[-1])
        
        return smoothed_path
