from __future__ import absolute_import
import time
import octoprint.plugin
from octoprint.events import Events
import serial
import threading
import re

class PFVSPlugin(octoprint.plugin.SettingsPlugin,
                 octoprint.plugin.AssetPlugin,
                 octoprint.plugin.TemplatePlugin,
                 octoprint.plugin.EventHandlerPlugin,
                 octoprint.plugin.OctoPrintPlugin):

    def __init__(self):
        super().__init__()
        self.is_filament_loading = False
        self.is_filament_unloading = False
        self.arduino_serial = None
        self.serial_thread = None
        self.running = False
        self.print_paused = False

    def on_after_startup(self):
        self._logger.info("PFVS Plugin initialized.")

    ##~~ SettingsPlugin mixin

    def get_settings_defaults(self):
        return {
            # Add plugin default settings if needed
        }

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

    ##~~ Event Handler Plugin

    def on_event(self, event, payload):
        self._logger.info(f"Event received: {event}")
        if event == "PrinterStateChanged":
            new_state = payload.get("state_id")
            self._logger.info(f"New printer state: {new_state}")
            
            if new_state == "PRINTING" and not self.print_paused:
                    self._logger.info("Detected state transition to PRINTING. Print is starting!")
                    self.print_paused = True
                    self._printer.pause_print()
                    self._logger.info("Print started - pausing for 30 seconds.")

                    # Start a thread
                    threading.Thread(target=self.delayed_resume_print, daemon=True).start()

    def delayed_resume_print(self):
        time.sleep(30)
        self._printer.resume_print()
        self.print_paused = False  # Reset flag
        self._logger.info("Resuming print after 30-second pause.")


    ##~~ G-code received hook

    def process_gcode(self, comm, line, *args, **kwargs):
        if "M701" in line:  # Filament loading command
            self.is_filament_loading = True
            self.is_filament_unloading = False
            self._logger.info("Filament is being loaded.")
        elif "M702" in line:  # Filament unloading command
            self.is_filament_loading = False
            self.is_filament_unloading = True
            self._logger.info("Filament is being unloaded.")
        else:
            self.is_filament_loading = False
            self.is_filament_unloading = False
            
        match = re.search(r'(\d+\.?\d*)/(\d+\.?\d*)', line)    
        if match:
            current_temp = float(match.group(1))  # Extract the current temperature
            target_temp = float(match.group(2))  # Extract the target temperature
    
            # Check if the current temperature is at least 95% of the target temperature
            if current_temp >= 0.95 * target_temp:
                self._printer.pause_print()
                self._logger.info(f"Print started - pausing for 30 seconds as temperature is {current_temp}/{target_temp} (>= 95%).")
                threading.Thread(target=self.delayed_resume_print, daemon=True).start()
            
        return line

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
