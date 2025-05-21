# Maze Solver & Arduino Code Generator

## Overview

This application processes uploaded maze images, finds the optimal path through the maze using the A* algorithm, and generates Arduino code to navigate a robot through the maze. The app features a streamlined web interface built with Streamlit, image processing capabilities with OpenCV, and code generation for Arduino robots.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The system follows a modular design pattern with clear separation of concerns:

1. **Web Interface**: Built with Streamlit to provide a simple, interactive UI for uploading maze images and displaying results.

2. **Image Processing**: Uses OpenCV to process maze images, detect paths, walls, and start/end points.

3. **Path Finding**: Implements the A* algorithm to find the optimal path through the maze.

4. **Code Generation**: Converts the path into Arduino or embedded C code that can be used to control a robot.

The application follows a pipeline architecture:
`Image Upload → Image Processing → Path Finding → Code Generation → Result Display`

## Key Components

### 1. MazeProcessor (`maze_processor.py`)

Handles the image processing pipeline to convert raw maze images into a binary representation suitable for pathfinding.

- Uses OpenCV for image manipulation
- Processes images through grayscale conversion, Gaussian blur, and binary thresholding
- Detects start and end points based on predefined color ranges (red for start, green for end)
- Creates a binary maze representation where 0=wall and 1=path

### 2. PathFinder (`path_finder.py`)

Implements the A* algorithm to find the optimal path through the processed maze.

- Uses the `pathfinding` library for A* implementation
- Converts the binary maze into a grid representation
- Applies path smoothing to optimize the route
- Returns both pixel coordinates and grid-based path information

### 3. Code Generators (`code_generator.py`)

Converts path information into executable code for robots.

- `ArduinoCodeGenerator`: Creates Arduino-compatible code with movement commands
- `EmbeddedCCodeGenerator`: Provides an alternative code generation format
- Translates path coordinates into directional commands (forward, turn left/right, etc.)
- Includes timing and delay parameters for robot execution

### 4. Streamlit Interface (`app.py`)

The main entry point and UI of the application.

- Provides a user-friendly interface for uploading maze images
- Displays processing results and generated code
- Orchestrates the flow between the different components of the system

## Data Flow

1. **Image Acquisition**: User uploads a maze image through the Streamlit interface.
2. **Image Processing**:
   - Convert to grayscale
   - Apply Gaussian blur to reduce noise
   - Apply binary thresholding to differentiate walls from paths
   - Detect start and end points using color detection
3. **Path Finding**:
   - Convert processed image to a grid representation
   - Apply A* algorithm to find the optimal path
   - Smooth the path to create more natural movements
4. **Code Generation**:
   - Convert path coordinates into robot movement commands
   - Generate Arduino-compatible code with proper syntax and structure
5. **Result Presentation**:
   - Display the original maze, processed maze, and the solution path
   - Present the generated code for the user to copy

## External Dependencies

The application relies on the following external libraries:

1. **Streamlit**: For the web interface and interactive elements
2. **OpenCV (cv2)**: For image processing and computer vision tasks
3. **NumPy**: For numerical operations and array manipulation
4. **Matplotlib**: For visualization
5. **Pillow (PIL)**: For additional image processing capabilities
6. **Pathfinding**: For implementing the A* algorithm

## Deployment Strategy

The application is configured to run as a Streamlit web application accessible through a web browser.

- Uses a Replit deployment configuration with autoscaling
- Port configuration set to 5000 for the web server
- Includes necessary dependencies in the `pyproject.toml` file
- Streamlit configuration ensures the app runs in headless mode on the specified port
- Dependencies are managed via the package manager with lock file for consistency

## Development Guidelines

1. **Adding New Features**:
   - Follow the modular design pattern
   - Add new image processing methods to `MazeProcessor`
   - Extend path finding algorithms in `PathFinder`
   - Create new code generator classes for different robot platforms

2. **Testing Strategy**:
   - Test with various maze images of different complexities
   - Verify path finding accuracy
   - Validate generated code format

3. **Code Organization**:
   - Keep component responsibilities clearly separated
   - Use meaningful variable and function names
   - Add appropriate documentation for complex algorithms