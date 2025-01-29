from collections import namedtuple

# Define the immutable scan object
MaterialScan = namedtuple("MaterialScan", ["name", "wavelengths", "intensities"])

# Example of defining known material scans
known_scans = [
    MaterialScan(
        name="PLA_Red",
        wavelengths=[410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940],
        intensities=[120, 200, 340, 150, 190, 280, 300, 400, 250, 180, 90, 60, 50, 70, 140, 100, 80, 60],
    ),
    MaterialScan(
        name="PETG_Blue",
        wavelengths=[410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940],
        intensities=[130, 220, 300, 200, 210, 260, 310, 420, 280, 170, 100, 70, 40, 60, 150, 110, 90, 50],
    ),
    # Add more scans as needed
]
