import cv2
import numpy as np

from solver import SolverRubik


class RubikScanner:
    def __init__(self):
        self.face = None
        self.colors = None
        self.faces = {
            "U": None,
            "D": None,
            "L": None,
            "R": None,
            "F": None,
            "B": None,
        }
        self.face_colors = {
            "U": None,
            "D": None,
            "L": None,
            "R": None,
            "F": None,
            "B": None,
        }
        self.current_face_index = 0
        self.face_order = ["U", "D", "L", "R", "F", "B"]
        self.processed_faces = {face: None for face in self.faces.keys()}
        self.solution_steps = []
        self.display_solution = False

    def capture_face(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Cannot open camera")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape
            rect_top_left = (width // 2 - 150, height // 2 - 150)
            rect_bottom_right = (width // 2 + 150, height // 2 + 150)
            cell_size = 100

            # Draw the rectangle
            cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 255, 0), 2)

            # Draw the grid
            for i in range(3):
                for j in range(3):
                    center_x = rect_top_left[0] + j * cell_size + cell_size // 2
                    center_y = rect_top_left[1] + i * cell_size + cell_size // 2
                    cv2.circle(frame, (center_x, center_y), 8, (255, 255, 255), -1)

            # Write text on the frame
            cv2.putText(
                frame,
                "Press 's' to save, 'n' for next face",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

            # Display the saved faces
            self.display_saved_faces()

            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1)
            if key == ord("s"):
                self.save_current_face(frame, height, width)
            elif key == ord("n"):
                if self.faces[self.face_order[self.current_face_index]] is not None:
                    if self.current_face_index < len(self.faces) - 1:
                        self.current_face_index += 1
                    else:
                        print("All faces saved. Solving...")

                        solver = SolverRubik(self.get_state())
                        solutions = solver.solve()
                        self.solution_steps = solutions.split()
                        self.display_solution = True
                        # Display the saved faces with the solution
                        self.display_saved_faces()
                else:
                    print("Please save the current face before moving to the next one.")
            elif key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def display_saved_faces(self):
        display_frame = np.zeros((600, 600, 3), dtype=np.uint8)

        for i, (face_name, face) in enumerate(self.faces.items()):
            if face is not None:
                row = i // 3
                col = i % 3
                y = row * 200
                x = col * 200
                if x + 200 > display_frame.shape[1] or y + 200 > display_frame.shape[0]:
                    continue
                processed_face = self.processed_faces[face_name]
                processed_face_resized = cv2.resize(processed_face, (200, 200))
                display_frame[y : y + 200, x : x + 200] = processed_face_resized  #
                cv2.putText(
                    display_frame,
                    face_name,
                    (x + 80, y + 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                self.draw_grid(display_frame[y : y + 200, x : x + 200])

        if self.display_solution:
            solution_text = " ".join(self.solution_steps)
            font_scale = 1.0
            max_width = display_frame.shape[1] - 20  # Padding
            while True:
                text_size = cv2.getTextSize(
                    solution_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                )[0]
                if text_size[0] <= max_width:
                    break
                font_scale -= 0.1
            text_x = (display_frame.shape[1] - text_size[0]) // 2
            text_y = display_frame.shape[0] - 30
            cv2.putText(
                display_frame,
                solution_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Saved Faces", display_frame)

    def draw_grid(self, face):
        cell_size = face.shape[0] // 3
        for i in range(1, 3):
            # Draw horizontal lines
            cv2.line(
                face, (0, i * cell_size), (face.shape[1], i * cell_size), (0, 0, 0), 1
            )
            # Draw vertical lines
            cv2.line(
                face, (i * cell_size, 0), (i * cell_size, face.shape[0]), (0, 0, 0), 1
            )

    def save_current_face(self, frame, height, width):
        self.face = frame[
            height // 2 - 150 : height // 2 + 150,
            width // 2 - 150 : width // 2 + 150,
        ]
        self.face = cv2.GaussianBlur(self.face, (5, 5), 0)
        face_name = self.face_order[self.current_face_index]
        self.faces[face_name] = self.face
        self.processed_faces[face_name], self.face_colors[face_name] = (
            self.process_face(self.face)
        )
        for row in self.face_colors[face_name]:
            row.reverse()

        print(f"Face {face_name} saved")

    def process_face(self, face):
        try:
            hsv_face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        except Exception as e:
            print(f"Error processing face: {e}")
            return face, None

        height, width, _ = face.shape
        cell_height, cell_width = height // 3, width // 3

        processed_face = np.zeros_like(face)
        face_colors = []

        for i in range(3):
            row_colors = []
            for j in range(3):
                cell = hsv_face[
                    i * cell_height : (i + 1) * cell_height,
                    j * cell_width : (j + 1) * cell_width,
                ]
                # Get samples from the center of the cell
                samples = []
                step = cell_height // 3
                for y in range(step // 2, cell_height, step):
                    for x in range(step // 2, cell_width, step):
                        samples.append(cell[y, x])
                samples = np.array(samples)

                # Get the median color
                median_color = np.median(samples, axis=0)
                color_name = self.get_color_name(median_color)

                row_colors.append(color_name)
                color_bgr = self.get_bgr_color(color_name)
                cv2.rectangle(
                    processed_face,
                    (j * cell_width, i * cell_height),
                    ((j + 1) * cell_width, (i + 1) * cell_height),
                    color_bgr,
                    -1,
                )
            face_colors.append(row_colors)

        self.draw_grid(processed_face)
        return processed_face, face_colors

    def get_color_name(self, hsv_color):
        colors = {
            "white": ([0, 0, 180], [180, 20, 255]),
            "orange": ([0, 50, 70], [9, 255, 255]),
            "red": ([170, 50, 70], [180, 255, 255]),
            "blue": ([90, 50, 70], [128, 255, 255]),
            "green": ([36, 50, 70], [89, 255, 255]),
            "yellow": ([10, 70, 100], [30, 255, 255]),
        }

        detected_color = "unknown"
        min_distance = float("inf")
        for color_name, (lower, upper) in colors.items():
            if self.in_range(hsv_color, lower, upper):
                return color_name

            distance = self.color_distance(hsv_color, lower, upper)
            if distance < min_distance:
                min_distance = distance
                detected_color = color_name

        return detected_color

    def get_bgr_color(self, color_name):
        color_dict = {
            "white": (255, 255, 255),
            "orange": (0, 165, 255),
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "unknown": (0, 0, 0),
        }
        return color_dict.get(color_name, (0, 0, 0))

    def in_range(self, hsv_color, lower, upper):
        return all(lower[i] <= hsv_color[i] <= upper[i] for i in range(3))

    def color_distance(self, hsv_color, lower, upper):
        midpoint = [(lower[i] + upper[i]) / 2 for i in range(3)]
        return np.linalg.norm(np.array(hsv_color) - np.array(midpoint))

    def get_state(self):
        return self.face_colors


def main():
    scanner = RubikScanner()
    scanner.capture_face()


if __name__ == "__main__":
    main()
