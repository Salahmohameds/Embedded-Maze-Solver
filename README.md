# Maze Path Finder & Arduino Code Generator

A Streamlit-based application that helps you solve mazes and generate Arduino code for robot navigation. The application can process maze images and generate corresponding Arduino code for robot movement.

## Features

- Upload and process maze images
- Extract yellow paths from maze images
- Solve mazes from scratch
- Generate Arduino code for robot navigation
- Support for command string input
- Visual path visualization
- Real-time code generation

## Requirements

- Python 3.11 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd MazePathFinder
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:5000)

3. Choose your input method:
   - Upload a maze image (JPG or PNG)
   - Enter a command string

4. For maze images:
   - Upload an image containing a maze
   - Choose between using the yellow path or solving the maze from scratch
   - View the generated path and Arduino code

5. For command strings:
   - Enter a formatted command string
   - View the generated Arduino code

## Project Structure

- `app.py`: Main Streamlit application
- `maze_processor.py`: Maze processing and path extraction
- `path_finder.py`: Maze solving algorithms
- `code_generator.py`: Arduino code generation
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Dependencies

- matplotlib: For visualization
- numpy: For numerical operations
- opencv-python: For image processing
- pathfinding: For maze solving algorithms
- pillow: For image handling
- streamlit: For the web interface

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 