from collections import namedtuple

# hi
# Define the immutable scan object
MaterialScan = namedtuple("MaterialScan", ["name", "wavelengths", "intensities", "settings"])

# Settings Arrays
PLA_SETTINGS = {
    "print_temp": 200,
    "bed_temp": 60,
    "fan_speed": 100,
    "print_speed": 50,
    "retraction_distance": 1.5,
    "retraction_speed": 35,
    "z_hop": 0.2,
    "flow_rate": 100,
    "infill_percentage": 20,
    "requires_enclosure": False,
}

PETG_SETTINGS = {
    "print_temp": 240,
    "bed_temp": 80,
    "fan_speed": 50,
    "print_speed": 40,
    "retraction_distance": 6.0,
    "retraction_speed": 25,
    "z_hop": 0.4,
    "flow_rate": 105,
    "infill_percentage": 30,
    "requires_enclosure": False,
}

ASA_SETTINGS = {
    "print_temp": 240,
    "bed_temp": 100,
    "fan_speed": 0,
    "print_speed": 50,
    "retraction_distance": 2.0,
    "retraction_speed": 30,
    "z_hop": 0.2,
    "flow_rate": 100,
    "infill_percentage": 20,
    "requires_enclosure": True,
}

# Predefined database of filament scans
known_scans = [
    MaterialScan(
        name="PLA_Red",
        wavelengths=[410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940],
        intensities=[120, 200, 340, 150, 190, 280, 300, 400, 250, 180, 90, 60, 50, 70, 140, 100, 80, 60],
        settings=PLA_SETTINGS,
    ),
    MaterialScan(
        name="PETG_Blue",
        wavelengths=[410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940],
        intensities=[130, 220, 300, 200, 210, 260, 310, 420, 280, 170, 100, 70, 40, 60, 150, 110, 90, 50],
        settings=PETG_SETTINGS,
    ),
    MaterialScan(
        name="ASA_Black",
        wavelengths=[410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940],
        intensities=[100, 180, 300, 250, 200, 230, 250, 300, 280, 200, 150, 100, 80, 70, 120, 140, 90, 60],
        settings=ASA_SETTINGS,
    ),
]
