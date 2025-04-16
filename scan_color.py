import numpy as np
import colorsys

# Define a set of known filament colors (RGB 0-255)
KNOWN_COLORS = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "Orange": (255, 165, 0),
    "Pink": (255, 192, 203),
    "Purple": (128, 0, 128),
    "Brown": (165, 42, 42),
    "Gray": (128, 128, 128),
    "Light Blue": (173, 216, 230),
    "Dark Green": (0, 100, 0),
    "Dark Red": (139, 0, 0),
    "Beige": (245, 245, 220)
}

# Function to compute Euclidean distance
def euclidean_distance(color1, color2):
    return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

# Function to convert RGB to HSV
def rgb_to_hsv(r, g, b):
    r_scaled, g_scaled, b_scaled = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_scaled, g_scaled, b_scaled)
    return h * 360, s * 100, v * 100  # Scale H to 0-360, S/V to 0-100

# Function to classify color
def classify_color(r, g, b):
    # Convert to HSV
    h, s, v = rgb_to_hsv(r, g, b)
    
    # Special Handling for Black & White
    if s < 10:  # Low saturation means it's grayscale
        if v > 85:
            return "White"
        elif v < 25:
            return "Black"
        else:
            return "Gray"

    # Find closest known color
    closest_color = min(KNOWN_COLORS, key=lambda color: euclidean_distance((r, g, b), KNOWN_COLORS[color]))
    
    return closest_color

# Example usage:
sample_colors = [
    (250, 10, 10),  # Should be classified as "Red"
    (230, 230, 230),  # Should be classified as "White"
    (10, 10, 10),  # Should be classified as "Black"
    (100, 200, 100),  # Should be classified as "Green"
    (200, 200, 0),  # Should be classified as "Yellow"
    (20, 150, 150),
    (25,25, 25)
]

for rgb in sample_colors:
    print(f"RGB {rgb} -> Classified as: {classify_color(*rgb)}")
