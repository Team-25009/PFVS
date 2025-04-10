from __future__ import absolute_import
import time
import octoprint.plugin
from octoprint.events import Events
import threading
import re
import sys
import os
import math
import numpy as np
from flask import jsonify
import RPi.GPIO as GPIO
from octoprint_pfvs import spectrometer as spect
from octoprint_pfvs.filament_gcodes import FILAMENTS
from octoprint_pfvs.predict_material import predict_material

class PFVSPlugin(octoprint.plugin.SettingsPlugin,
                 octoprint.plugin.AssetPlugin,
                 octoprint.plugin.TemplatePlugin,
                 octoprint.plugin.EventHandlerPlugin,
                 octoprint.plugin.BlueprintPlugin,
                 octoprint.plugin.OctoPrintPlugin):

    def __init__(self):
        super().__init__()
        self.is_filament_loading = False
        self.is_filament_unloading = False
        self.print_paused = False
        self.print_starting = False
        self.spectrometer_thread = None
        self.spectrometer_running = False 
        self.predicted_material = ""

    def on_after_startup(self):
        self._logger.info("PFVS Plugin initialized.")
        try:
            spect.init()
            self._logger.info("Spectrometer initialized successfully.")
        except Exception as e:
            self._logger.error(f"Failed to initialize spectrometer: {e}")

    ##~~ SettingsPlugin mixin

    def get_settings_defaults(self):
        return {}

    ##~~ AssetPlugin mixin

    def get_assets(self):
        return {
            "js": ["js/pfvs.js"],
            "css": ["css/pfvs.css"],
            "less": ["less/pfvs.less"]
        }

    ##~~ TemplatePlugin mixin

    def get_template_vars(self):
        return {"plugin_version": self._plugin_version}
    
    ##~~ Template Plugin Mixin
    def get_template_configs(self):
        return [
            {
                "type": "tab", 
                "name": "PFVS",
                "template": "pfvs.jinja2",
            },
            {
            "type": "generic",
            "template": None,
            }
        ]

    ##~~ Event Handler Plugin

    def on_event(self, event, payload):
        self._logger.info(f"Event received: {event}")
        
        if event == "PrinterStateChanged":
            new_state = payload.get("state_id")
            self._logger.info(f"New printer state: {new_state}")

            if new_state == "STARTING":
                self._logger.info("Print has officially started.")
                self.print_starting = True  # Set flag when print starts

    def delayed_resume_print(self):
        time.sleep(30)
        self._printer.resume_print()
        self.print_paused = False  
        self._logger.info("Resuming print after 30-second pause.")

    ##~~ G-code received hook

    def process_gcode(self, comm, line, *args, **kwargs):
        """ Processes received G-code and handles filament verification & temperature adjustments """

        if "M701" in line:  # Filament load command detected
            self.is_filament_loading = True
            self.is_filament_unloading = False
            self._logger.info("Filament is being loaded.") # Check if filament is present
                # Run spectrometer scan
            self.filament_scan()
            self._logger.info("Filament is loaded and scan happened")
            self._logger.info(f"Predicted material: {self.predicted_material}") 
            self._plugin_manager.send_plugin_message(
                self._identifier, 
                {"predicted_material": self.predicted_material}
            )

        elif "M702" in line:  # Filament unload command detected
            self.is_filament_loading = False
            self.is_filament_unloading = True
            self._logger.info("Filament is being unloaded.")

        else:
            self.is_filament_loading = False
            self.is_filament_unloading = False
            
        match = re.search(r'(\d+\.?\d*)/(\d+\.?\d*)', line)
        current_temp = float(match.group(1))  
        target_temp = float(match.group(2))    
        if match and self.print_starting and target_temp != 170.0:
            # Check if we are close to the target temperature
            if current_temp >= 0.95 * target_temp:
                self.filament_scan()
                self._logger.info(f"Predicted material: {self.predicted_material}") 
                self._plugin_manager.send_plugin_message(
                    self._identifier, 
                    {"predicted_material": self.predicted_material}
                )
                if self.predicted_material == "ASA":
                    self._logger.info(f"Cannot print ASA on Prusa Mini")
                    self._printer.cancel_print()
                    return line

                # If target temperature is incorrect, adjust it first
                if not math.isclose(target_temp, filament.print_temp, rel_tol=1e-2):  
                    self._logger.info(f"Incorrect target temperature detected: {target_temp}°C. Changing to {filament.print_temp}°C.")
                    if self.predicted_material in FILAMENTS: 
                        filament = FILAMENTS[self.predicted_material]

                        self._logger.info(f"Current temp: {current_temp}°C | Expected temp: {filament.print_temp}°C")
                        self._logger.info("Temperature mismatch detected. Sending new G-code settings...")
                        gcode_commands = filament.generate_gcode()
                        self._printer.commands(gcode_commands)
                        self._logger.info(f"Sent updated G-code commands: {gcode_commands}")
                    else:
                        self._logger.warning(f"Unknown filament type: {self.predicted_material}. No preset settings found.")

                # Start a separate thread to resume after 30 seconds
                threading.Thread(target=self.delayed_resume_print, daemon=True).start()    

        return line


    ##~~ Spectrometer Handling
    def is_filament_detected(self):
        """Returns True if the IR sensor detects filament."""
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        return GPIO.input(11) == GPIO.HIGH

    def filament_scan(self):
        try:
            spect.setGain(3)
            spect.setIntegrationTime(63)
            spect.shutterLED("AS72651", False)
            spect.shutterLED("AS72652", False)
            spect.shutterLED("AS72653", False)
            time.sleep(0.18)
            dark_spect_data = spect.readRAW()
            self._logger.info(f"Raw Dark Spectrometer Data: {dark_spect_data}")
            time.sleep(1.0)  

            spect.shutterLED("AS72651", True)
            spect.shutterLED("AS72652", True)
            spect.shutterLED("AS72653", True)
            # Reading spectrometer data
            time.sleep(0.18)
            light_spect_data = spect.readRAW()
            time.sleep(0.18)
            light_spect_data = spect.readRAW()
            time.sleep(0.18)
            light_spect_data = spect.readRAW()
             

            for i in range(len(light_spect_data)):
                    light_spect_data[i] = light_spect_data[i] - dark_spect_data[i]
                
                # Finally, pass the spectrometer data to the prediction function
            self._logger.info(f"Raw Spectrometer Data: {light_spect_data}")
            self.predicted_material = predict_material(light_spect_data, 'R')
            self._logger.info(f"Predicted material: {self.predicted_material}") 
            time.sleep(1)  # Adjust sampling rate
        except Exception as e:
            self._logger.error(f"Error reading spectrometer data: {e}")
            
    def start_spectrometer(self):
        """Starts a separate thread for reading spectrometer data."""
        if self.spectrometer_running:
            self._logger.info("Spectrometer is already running.")
            return
        
        if not self.is_filament_detected():
            self._logger.warning("No filament detected. Spectrometer will not start.")
            return
        
        # Add if statement to see if there is filament detected first before running a scan
        self.spectrometer_running = True
        self.spectrometer_thread = threading.Thread(target=self.read_spectrometer_data, daemon=True)
        self.spectrometer_thread.start()
        self._logger.info("Spectrometer data collection started.")

    def stop_spectrometer(self):
        """Stops the spectrometer thread."""
        spect.shutterLED("AS72651", False)
        spect.shutterLED("AS72652", False)
        spect.shutterLED("AS72653", False)
        self.spectrometer_running = False
        self._logger.info("Stopping spectrometer data collection.")

    def read_spectrometer_data(self):
        """Reads data from the spectrometer and sends it to the web interface."""
        try:
            spect.setGain(3)
            spect.setIntegrationTime(63)
            spect.shutterLED("AS72651", False)
            spect.shutterLED("AS72652", False)
            spect.shutterLED("AS72653", False)
            time.sleep(0.18)
            dark_spect_data = spect.readRAW()
            self._logger.info(f"Raw Dark Spectrometer Data: {dark_spect_data}")
            time.sleep(1.0)  

            spect.shutterLED("AS72651", True)
            spect.shutterLED("AS72652", True)
            spect.shutterLED("AS72653", True)
            while self.spectrometer_running:
                # Reading spectrometer data
                time.sleep(0.18)
                light_spect_data = spect.readRAW() 

                for i in range(len(light_spect_data)):
                    light_spect_data[i] = light_spect_data[i] - dark_spect_data[i]
                
                # Finally, pass the spectrometer data to the prediction function
                self._logger.info(f"Raw Spectrometer Data: {light_spect_data}")
                predicted_material = predict_material(light_spect_data, 'R')
                self._logger.info(f"Predicted material: {predicted_material}")

                # Send data to web UI
                self._plugin_manager.send_plugin_message(
                    self._identifier, 
                    {"spectrometer_data": light_spect_data, "predicted_material": predicted_material}
                )
                
                time.sleep(1)  # Adjust sampling rate
        except Exception as e:
            self._logger.error(f"Error reading spectrometer data: {e}")

    ##~~ Software update hook

    def get_update_information(self):
        return {
            "pfvs": {
                "displayName": "PFVS Plugin",
                "displayVersion": self._plugin_version,
                "type": "github_release",
                "user": "samnperry",
                "repo": "PFVS",
                "current": self._plugin_version,
                "pip": "https://github.com/samnperry/PFVS/archive/{target_version}.zip",
            }
        }

    ##~~ API for UI to Start/Stop Spectrometer

    @octoprint.plugin.BlueprintPlugin.route("/start_spectrometer", methods=["POST"])
    def api_start_spectrometer(self):
        """API endpoint to start spectrometer via UI."""
        self.start_spectrometer()
        return jsonify(status="Spectrometer started")

    @octoprint.plugin.BlueprintPlugin.route("/stop_spectrometer", methods=["POST"])
    def api_stop_spectrometer(self):
        """API endpoint to stop spectrometer via UI."""
        self.stop_spectrometer()
        return jsonify(status="Spectrometer stopped")

__plugin_name__ = "PFVS Plugin"
__plugin_pythoncompat__ = ">=3,<4"

def __plugin_load__():
    global __plugin_implementation__
    __plugin_implementation__ = PFVSPlugin()

    global __plugin_hooks__
    __plugin_hooks__ = {
        "octoprint.plugin.softwareupdate.check_config": __plugin_implementation__.get_update_information,
        "octoprint.comm.protocol.gcode.received": (__plugin_implementation__.process_gcode, 1),
    }
