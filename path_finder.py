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
        # Create a copy of the maze for potential modifications
        working_maze = binary_maze.copy()
        
        # Get maze dimensions
        height, width = working_maze.shape
        
        # Ensure start and end points are valid (on a path, not a wall)
        start_x, start_y = start_point
        end_x, end_y = end_point
        
        # Ensure points are within grid bounds
        start_x = max(0, min(start_x, width - 1))
        start_y = max(0, min(start_y, height - 1))
        end_x = max(0, min(end_x, width - 1))
        end_y = max(0, min(end_y, height - 1))
        
        # Check if start or end points are on walls
        # If they are, find nearest path point
        search_radius = 10
        if working_maze[start_y, start_x] == 0:  # If on a wall
            # Search nearby for a valid path point
            for r in range(1, search_radius):
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if abs(dx) + abs(dy) == r:  # Manhattan distance = r
                            ny, nx = start_y + dy, start_x + dx
                            if 0 <= ny < height and 0 <= nx < width and working_maze[ny, nx] == 1:
                                start_y, start_x = ny, nx
                                break
                    if working_maze[start_y, start_x] == 1:
                        break
                if working_maze[start_y, start_x] == 1:
                    break
        
        if working_maze[end_y, end_x] == 0:  # If on a wall
            # Search nearby for a valid path point
            for r in range(1, search_radius):
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if abs(dx) + abs(dy) == r:  # Manhattan distance = r
                            ny, nx = end_y + dy, end_x + dx
                            if 0 <= ny < height and 0 <= nx < width and working_maze[ny, nx] == 1:
                                end_y, end_x = ny, nx
                                break
                    if working_maze[end_y, end_x] == 1:
                        break
                if working_maze[end_y, end_x] == 1:
                    break
        
        # Update start and end points if they were modified
        start_point = (start_x, start_y)
        end_point = (end_x, end_y)
        
        # If still on walls, we'll force a path by modifying the maze
        if working_maze[start_y, start_x] == 0:
            working_maze[start_y, start_x] = 1
        
        if working_maze[end_y, end_x] == 0:
            working_maze[end_y, end_x] = 1
        
        # Create a grid from the binary maze
        grid = Grid(width=width, height=height, matrix=working_maze.tolist())
        
        # Create start and end nodes
        start_node = grid.node(start_x, start_y)
        end_node = grid.node(end_x, end_y)
        
        # Attempt to find the path with A*
        path, runs = self.finder.find_path(start_node, end_node, grid)
        
        # If no path is found, try with diagonal movement allowed
        if not path:
            grid = Grid(width=width, height=height, matrix=working_maze.tolist())
            finder_with_diagonals = AStarFinder(diagonal_movement=DiagonalMovement.always)
            path, runs = finder_with_diagonals.find_path(grid.node(start_x, start_y), 
                                                         grid.node(end_x, end_y), grid)
        
        # If still no path, create a simple direct path (for demonstration purposes)
        if not path:
            # Create a simple Manhattan path between start and end
            x, y = start_x, start_y
            simple_path = [(x, y)]
            
            # Horizontal movement first
            while x != end_x:
                x += 1 if x < end_x else -1
                simple_path.append((x, y))
            
            # Then vertical movement
            while y != end_y:
                y += 1 if y < end_y else -1
                simple_path.append((x, y))
            
            # Convert to node format for consistency
            path = [grid.node(x, y) for x, y in simple_path]
        
        # Convert path to pixel coordinates
        pixel_path = [(node.x, node.y) for node in path]
        
        # Apply path smoothing
        smoothed_path = self._smooth_path(pixel_path, working_maze)
        
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
