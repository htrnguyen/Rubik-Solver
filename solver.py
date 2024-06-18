import kociemba


class SolverRubik:
    def __init__(self, state=None):
        self.rubik_state = state
        self.solved_state = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
        self.color_map = {
            "white": "U",
            "yellow": "D",
            "green": "L",
            "blue": "R",
            "red": "F",
            "orange": "B",
        }

    def rubik_to_string(self, state):
        order = "URFDLB"
        result = []
        for face in order:
            for row in state[face]:
                for color in row:
                    result.append(self.color_map[color])
        return "".join(result)

    def count_colors(self, cube_string):
        colors = "URFDLB"
        count = {color: 0 for color in colors}
        for char in cube_string:
            if char in colors:
                count[char] += 1
        return count

    def solve(self):
        if self.rubik_state is None:
            return "No state provided"

        rubik_string = self.rubik_to_string(self.rubik_state)
        color_count = self.count_colors(rubik_string)

        if rubik_string == self.solved_state:
            return "Rubik's cube is already solved"

        for color, count in color_count.items():
            if count != 9:
                return (
                    f"Error: Color {color} appears {count} times, should appear 9 times"
                )

        try:
            solution = kociemba.solve(rubik_string)
            return solution
        except ValueError as e:
            return f"Error: {e}"
