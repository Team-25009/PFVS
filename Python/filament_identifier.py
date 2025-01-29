import numpy as np

class FilamentIdentifier:
    def __init__(self, known_scans):
        self.known_scans = known_scans

    def identify(self, wavelengths, intensities):
        # Ensure the input wavelengths align with known scans
        assert len(wavelengths) == len(self.known_scans[0].wavelengths)

        best_match = None
        smallest_difference = float("inf")

        # Compare the new scan against each known scan
        for scan in self.known_scans:
            # Calculate the sum of absolute differences between intensities
            difference = np.sum(np.abs(np.array(scan.intensities) - np.array(intensities)))

            if difference < smallest_difference:
                smallest_difference = difference
                best_match = scan.name

        return best_match
