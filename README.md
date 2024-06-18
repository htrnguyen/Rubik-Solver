# Rubik's Cube Scanner and Solver

## Requirements

-   Python 3.8 or higher
-   OpenCV
-   NumPy
-   Kociemba

## Installation

1. Install the necessary libraries by running the following command:

```bash
pip install opencv-python numpy kociemba
```

## Usage

1. Run the program using the following command:

```bash
python rubik.py
```

2. The program will open the camera, and you can scan each face of the Rubik's Cube. When the cube face is centered on the screen, press 's' to save the colors of the current face. Press 'n' to move to the next face.

![Scan Rubik](/Document/Scan_Rubik.png)

3. After scanning all the faces, the program will display the steps to solve the Rubik's Cube.

![Solution](/Document/Solution.png)

### Demo

[![Video Demo](https://img.youtube.com/vi/7rKNxRrDnnY/maxresdefault.jpg)](https://youtube.com/watch?v=7rKNxRrDnnY)