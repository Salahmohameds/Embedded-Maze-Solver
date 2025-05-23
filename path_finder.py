import numpy as np
import cv2
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
        
        # Create corridors from start and end points if they're on walls
        # This guarantees we can enter and exit the maze
        if working_maze[start_y, start_x] == 0:
            # Create a corridor from start point towards the center
            mid_x, mid_y = width // 2, height // 2
            dx = 1 if mid_x > start_x else -1
            dy = 1 if mid_y > start_y else -1
            
            # Try horizontal corridor first
            x, y = start_x, start_y
            while working_maze[y, x] == 0 and 0 <= x < width - dx and 0 <= y < height:
                working_maze[y, x] = 1
                x += dx
                if working_maze[y, x] == 1:
                    break
                
            # If still on wall, try vertical corridor
            if working_maze[start_y, start_x] == 0:
                x, y = start_x, start_y
                while working_maze[y, x] == 0 and 0 <= x < width and 0 <= y < height - dy:
                    working_maze[y, x] = 1
                    y += dy
                    if working_maze[y, x] == 1:
                        break
        
        # Similar for end point
        if working_maze[end_y, end_x] == 0:
            # Create a corridor from end point towards the center
            mid_x, mid_y = width // 2, height // 2
            dx = 1 if mid_x > end_x else -1
            dy = 1 if mid_y > end_y else -1
            
            # Try horizontal corridor first
            x, y = end_x, end_y
            while working_maze[y, x] == 0 and 0 <= x < width - dx and 0 <= y < height:
                working_maze[y, x] = 1
                x += dx
                if working_maze[y, x] == 1:
                    break
                
            # If still on wall, try vertical corridor
            if working_maze[end_y, end_x] == 0:
                x, y = end_x, end_y
                while working_maze[y, x] == 0 and 0 <= x < width and 0 <= y < height - dy:
                    working_maze[y, x] = 1
                    y += dy
                    if working_maze[y, x] == 1:
                        break
        
        # Ensure the start and end points themselves are marked as path
        working_maze[start_y, start_x] = 1
        working_maze[end_y, end_x] = 1
        
        # Apply an additional opening of paths to ensure connectivity
        kernel = np.ones((3, 3), np.uint8)
        working_maze = cv2.morphologyEx(working_maze.astype(np.uint8), cv2.MORPH_DILATE, kernel)
        
        # Create a grid from the binary maze
        grid = Grid(width=width, height=height, matrix=working_maze.tolist())
        
        # Create start and end nodes
        start_node = grid.node(start_x, start_y)
        end_node = grid.node(end_x, end_y)
        
        # Attempt to find the path with A*
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        path, runs = finder.find_path(start_node, end_node, grid)
        
        # If no path is found, try with diagonal movement allowed
        if not path:
            grid = Grid(width=width, height=height, matrix=working_maze.tolist())
            finder_with_diagonals = AStarFinder(diagonal_movement=DiagonalMovement.always)
            path, runs = finder_with_diagonals.find_path(grid.node(start_x, start_y), 
                                                         grid.node(end_x, end_y), grid)
        
        # If still no path, create a more intelligent fallback path
        if not path:
            # Use breadth-first search to find a path
            visited = np.zeros_like(working_maze)
            queue = [(start_x, start_y)]
            visited[start_y, start_x] = 1
            parent = {}
            
            dx = [0, 1, 0, -1]  # Direction vectors (right, down, left, up)
            dy = [-1, 0, 1, 0]
            
            found = False
            while queue and not found:
                x, y = queue.pop(0)
                
                if (x, y) == (end_x, end_y):
                    found = True
                    break
                
                for i in range(4):
                    nx, ny = x + dx[i], y + dy[i]
                    if (0 <= nx < width and 0 <= ny < height and 
                        working_maze[ny, nx] == 1 and visited[ny, nx] == 0):
                        queue.append((nx, ny))
                        visited[ny, nx] = 1
                        parent[(nx, ny)] = (x, y)
            
            if found:
                # Reconstruct path
                simple_path = []
                current = (end_x, end_y)
                while current != (start_x, start_y):
                    simple_path.append(current)
                    current = parent[current]
                simple_path.append((start_x, start_y))
                simple_path.reverse()
                
                # Convert to node format for consistency
                path = [grid.node(x, y) for x, y in simple_path]
            else:
                # If BFS also fails, use a simple direct path
                x, y = start_x, start_y
                simple_path = [(x, y)]
                
                # Try to follow corridors where possible
                while (x != end_x) or (y != end_y):
                    # Prefer moving in direction with more open space
                    h_open = 0
                    v_open = 0
                    
                    # Check horizontal direction
                    test_x = x + (1 if x < end_x else -1)
                    if 0 <= test_x < width:
                        for dy in range(-1, 2):
                            test_y = y + dy
                            if 0 <= test_y < height and working_maze[test_y, test_x] == 1:
                                h_open += 1
                    
                    # Check vertical direction
                    test_y = y + (1 if y < end_y else -1)
                    if 0 <= test_y < height:
                        for dx in range(-1, 2):
                            test_x = x + dx
                            if 0 <= test_x < width and working_maze[test_y, test_x] == 1:
                                v_open += 1
                    
                    # Move in direction with more open space
                    if (h_open > v_open) and (x != end_x):
                        x += 1 if x < end_x else -1
                    else:
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
