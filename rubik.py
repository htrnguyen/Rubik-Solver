import cv2
import numpy as np
from sklearn.cluster import KMeans

class RubikScanner:
    """
    Class for scanning a Rubik's cube and extracting the colors of the stickers
    """

    def __init__(self):
        self.face = None
        self.colors = None

    def capture_face(self, camera_id=0):
        """
        Capture the image from the camera
        """
        cap = cv2.VideoCapture(camera_id)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape
            rectangle = (
                frame,
                (width // 2 - 100, height // 2 - 100),
                (width // 2 + 100, height // 2 + 100),
                (0, 255, 0),
                2,
            )

            cv2.rectangle(*rectangle)

            cv2.imshow("Press 'q' to save", frame)
            cv2.imshow(
                "Face",
                frame[
                    height // 2 - 100 : height // 2 + 100,
                    width // 2 - 100 : width // 2 + 100,
                ],
            )

            if cv2.waitKey(1) == ord("q"):
                # cv2.imwrite(
                #     "face.jpg",
                #     frame[
                #         height // 2 - 100 : height // 2 + 100,
                #         width // 2 - 100 : width // 2 + 100,
                #     ],
                # )
                self.face = frame[
                    height // 2 - 100 : height // 2 + 100,
                    width // 2 - 100 : width // 2 + 100,
                ]
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_face(self):
        """
        Process the face image to extract the colors of the stickers
        """
        if self.face is None:
            print("No face image captured.")
            return

        # Chuyển đổi hình ảnh sang không gian màu HSV
        hsv_face = cv2.cvtColor(self.face, cv2.COLOR_BGR2HSV)

        # Kích thước của mỗi ô
        height, width, _ = self.face.shape
        cell_height, cell_width = height // 3, width // 3

        self.colors = []

        # Chia hình ảnh thành 9 ô nhỏ và lấy mẫu từ nhiều điểm
        for i in range(3):
            row_colors = []
            for j in range(3):
                cell = hsv_face[
                    i * cell_height : (i + 1) * cell_height,
                    j * cell_width : (j + 1) * cell_width,
                ]

                # Lấy mẫu từ nhiều điểm trong mỗi ô
                samples = []
                step = cell_height // 3
                for y in range(step // 2, cell_height, step):
                    for x in range(step // 2, cell_width, step):
                        samples.append(cell[y, x])
                samples = np.array(samples)

                # Tính giá trị trung bình của các mẫu
                avg_color = np.mean(samples, axis=0)
                row_colors.append(avg_color)
            self.colors.append(row_colors)

        # Lưu màu của từng ô
        for row in self.colors:
            print([self.get_color_name(color) for color in row])

    def get_color_name(self, hsv_color):
        """
        Convert HSV color to a human-readable color name
        """
        colors = {
            "white": ([0, 0, 200], [180, 20, 255]),
            "yellow": ([25, 100, 100], [35, 255, 255]),
            "red1": ([0, 50, 70], [9, 255, 255]),
            "red2": ([170, 50, 70], [180, 255, 255]),
            "blue": ([90, 50, 70], [128, 255, 255]),
            "green": ([36, 50, 70], [89, 255, 255]),
            "orange": ([10, 100, 100], [24, 255, 255]),
        }

        detected_color = "unknown"
        min_distance = float('inf')
        for color_name, (lower, upper) in colors.items():
            if self.in_range(hsv_color, lower, upper):
                return color_name

            # Calculate the Euclidean distance to each color
            distance = self.color_distance(hsv_color, lower, upper)
            if distance < min_distance:
                min_distance = distance
                detected_color = color_name

        return detected_color

    def in_range(self, hsv_color, lower, upper):
        """
        Check if an HSV color is within the given range
        """
        return all(lower[i] <= hsv_color[i] <= upper[i] for i in range(3))

    def color_distance(self, hsv_color, lower, upper):
        """
        Calculate the Euclidean distance between a color and the range midpoint
        """
        midpoint = [(lower[i] + upper[i]) / 2 for i in range(3)]
        return np.linalg.norm(np.array(hsv_color) - np.array(midpoint))


def main():
    scanner = RubikScanner()
    scanner.capture_face()
    scanner.process_face()

if __name__ == "__main__":
    main()
