import serial
import time
from filament_database import known_scans
from filament_identifier import FilamentIdentifier
from prusa_interface import PrusaMiniInterface  # Optional: for applying printer settings

# Configuration
arduino_port = "COM3"  # Replace with your Arduino's port
baud_rate = 115200
reset_delay = 2  # Delay after resetting the Arduino (in seconds)

def reset_arduino(arduino):
    """
    Resets the Arduino by toggling the DTR (Data Terminal Ready) line.
    """
    print("Resetting Arduino...")
    arduino.setDTR(False)
    time.sleep(0.5)
    arduino.setDTR(True)
    time.sleep(reset_delay)  # Wait for Arduino to initialize

def wait_for_ready_signal(arduino):
    """
    Waits for the Arduino to send a ready signal indicating it is ready to receive commands.
    """
    print("Waiting for Arduino to signal readiness...")
    while True:
        response = arduino.readline().decode().strip()
        print(f"Raw response: {response}")  # Debug output
        if "Ready to receive scan requests..." in response:
            print("Arduino is ready.")
            break

def get_scan_from_arduino(arduino):
    """
    Sends a request to the Arduino to perform a scan and retrieves the spectral data.
    """
    print("Requesting scan from Arduino...")
    arduino.write(b"GET_SCAN\n")  # Send command to Arduino
    while True:
        response = arduino.readline().decode().strip()
        print(f"Raw response: {response}")  # Debug output

        # Look for a valid spectral data response (18 numeric values separated by " | ")
        if response.count(" | ") == 17:  # 17 separators mean 18 values
            try:
                # Parse response into a list of floats
                sensor_data = [float(value) for value in response.split(" | ")]
                return sensor_data
            except ValueError:
                print("Failed to parse sensor data.")
                return None
        elif response == "Invalid command. Use 'GET_SCAN' to trigger a scan.":
            print("Arduino received an invalid command.")
        else:
            print("Skipping non-data line.")

def identify_filament(sensor_data):
    """
    Identifies the filament type based on the scan data using the FilamentIdentifier.
    """
    # Wavelengths corresponding to the AS7265x sensor
    wavelengths = [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940]

    # Initialize filament identifier
    identifier = FilamentIdentifier(known_scans)

    # Identify the filament
    filament_name = identifier.identify(wavelengths, sensor_data)
    return filament_name

def main():
    try:
        # Open serial connection to Arduino
        arduino = serial.Serial(port=arduino_port, baudrate=baud_rate, timeout=2)
        reset_arduino(arduino)

        # Wait for the Arduino to indicate it's ready
        wait_for_ready_signal(arduino)

        # Fetch spectral data from Arduino
        sensor_data = get_scan_from_arduino(arduino)

        if sensor_data:
            print("Scan data received:", sensor_data)

            # Identify filament
            print("Identifying filament...")
            filament_name = identify_filament(sensor_data)
            if filament_name:
                print(f"Identified filament: {filament_name}")

                # Fetch the MaterialScan object from known_scans
                filament = next((scan for scan in known_scans if scan.name == filament_name), None)
                if filament is not None:
                    print(f"Filament settings: {filament.settings}")

                    # Optional: Apply settings to Prusa Mini
                    apply_settings = input("Apply these settings to the printer? (y/n): ").strip().lower()
                    if apply_settings == "y":
                        prusa = PrusaMiniInterface(port="/dev/ttyUSB0")  # Replace with your Prusa's port
                        prusa.set_printer_settings(filament.settings)
                        prusa.close()
                else:
                    print("Error: Filament found in database but settings could not be retrieved.")
            else:
                print("No matching filament found.")
        else:
            print("Failed to retrieve scan data.")

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    finally:
        if 'arduino' in locals() and arduino.is_open:
            arduino.close()
            print("Arduino connection closed.")

if __name__ == "__main__":
    main()
