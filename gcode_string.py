def pla_generate_gcode_basic():

    # Generates a basic list of G-code commands based on PLA settings.
   
    return [
        f"M104 S{200} ; Set hotend temperature",
        f"M140 S{60} ; Set bed temperature",
        f"M106 S{255} ; Set fan speed (0-255 scale)",
        f"M220 S{50} ; Set print speed (percentage)",
        f"M207 S{1.5} F{2100} ; Set retraction distance and speed",
        f"M204 S{0.2} ; Set Z-hop height",
        f"M221 S{100} ; Set extrusion flow rate",
        f"M486 S{20} ; Set infill percentage",
        "M500 ; Save settings",
    ]
    
def petg_generate_gcode_basic():

    # Generates a basic list of G-code commands based on PETG settings.
   
    return [
        f"M104 S{240} ; Set hotend temperature",
        f"M140 S{80} ; Set bed temperature",
        f"M106 S{128} ; Set fan speed (0-255 scale)",
        f"M220 S{40} ; Set print speed (percentage)",
        f"M207 S{6.0} F{1500} ; Set retraction distance and speed*60",
        f"M204 S{0.4} ; Set Z-hop height",
        f"M221 S{105} ; Set extrusion flow rate",
        f"M486 S{30} ; Set infill percentage",
        "M500 ; Save settings",
    ]
    
def asa_generate_gcode_basic():

    # Generates a basic list of G-code commands based on ASA settings.
   
    return [
        f"M104 S{260} ; Set hotend temperature",
        f"M140 S{100} ; Set bed temperature",
        f"M106 S{0} ; Set fan speed*2.55 (0-255 scale)",
        f"M220 S{50} ; Set print speed (percentage)",
        f"M207 S{2.0} F{1800} ; Set retraction distance and speed*60",
        f"M204 S{0.2} ; Set Z-hop height",
        f"M221 S{100} ; Set extrusion flow rate",
        f"M486 S{20} ; Set infill percentage",
        "M500 ; Save settings",
    ]