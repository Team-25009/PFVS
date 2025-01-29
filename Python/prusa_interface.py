import serial

class PrusaMiniInterface:
    def __init__(self, port, baudrate=115200):
        # Initialize the serial connection
        self.serial_conn = serial.Serial(port, baudrate, timeout=2)
        print("Connected to Prusa Mini.")

    def send_gcode(self, command):
        # Send a G-code command to the printer
        if not command.endswith("\n"):
            command += "\n"
        self.serial_conn.write(command.encode())
        response = self.serial_conn.readline().decode().strip()
        print(f"> {command.strip()} \n< {response}")
        return response

    def set_printer_settings(self, settings):
        # Apply printer settings via G-code
        self.send_gcode(f"M104 S{settings['print_temp']}")  # Set extruder temp
        self.send_gcode(f"M140 S{settings['bed_temp']}")    # Set bed temp
        self.send_gcode(f"M106 S{int(settings['fan_speed'] * 255 / 100)}")  # Set fan speed
        self.send_gcode(f"M220 S{settings['print_speed']}") # Set print speed multiplier
        self.send_gcode(f"M207 S{settings['retraction_distance']} F{settings['retraction_speed'] * 60}")  # Set retraction
        self.send_gcode(f"M221 S{settings['flow_rate']}")   # Set flow rate
        self.send_gcode("M500")  # Save settings to EEPROM (optional)
        print("Settings applied successfully.")

    def close(self):
        # Close the serial connection
        self.serial_conn.close()
        print("Connection closed.")
