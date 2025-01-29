from filament_database import known_scans
from filament_identifier import FilamentIdentifier

# Test samples with their names for easier identification
test_samples = [
    {
        "name": "RedPLA_Light",
        "intensities": [
            495.01, 919.49, 2218.19, 252.01, 1136.6, 607.06, 57.94, 182.3, 3271.24,
            284.13, 2787.93, 28.75, 384.23, 215.53, 736.54, 1825.16, 57.46, 34.58
        ],
    },
    {
        "name": "BlackPLA_Light",
        "intensities": [
            114.93, 751.51, 1891.4, 247.62, 1032, 533.93, 53.35, 174.01, 384.78,
            224.44, 75.32, 21.8, 39.8, 23.95, 62.51, 69.72, 38.95, 31.77
        ],
    },
    {
        "name": "BluePLA_Light",
        "intensities": [
            326.73, 833.07, 2054.33, 287.13, 1099.91, 597.12, 61.95, 197.21, 388.4,
            240.67, 276.52, 24.58, 99.1, 65.24, 176.66, 600.52, 45.33, 34.58
        ],
    },
    {
        "name": "GreenPLA_Light",
        "intensities": [
            183.89, 816.57, 2054.33, 270.45, 1212.32, 614.87, 58.51, 177.88, 460.77,
            234.94, 203.27, 22.72, 69.86, 78.45, 110.53, 203.55, 45.33, 32.71
        ],
    },
]

# Initialize the filament identifier with the known scans database
identifier = FilamentIdentifier(known_scans)

# Iterate through each test sample and identify it
for sample in test_samples:
    print(f"Testing sample: {sample['name']}")
    identified_material = identifier.identify(
        wavelengths=[410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940],
        intensities=sample["intensities"],
    )
    print(f"Identified Material: {identified_material}\n")
