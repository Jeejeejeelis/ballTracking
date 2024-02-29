
class Color:
    def __init__(self):
        self.colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "gray": (128, 128, 128),
            "light_blue": (255, 191, 0),
            "purple": (128, 0, 128),
            "brown": (0, 0, 128),
            "orange": (0, 165, 255),
            "pink": (147, 20, 255)
        }

    def get(self, color_name):
        return self.colors.get(color_name, (0, 0, 0))  # default to black if color_name is not found