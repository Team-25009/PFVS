#include <Wire.h>
#include <SparkFun_AS7265X.h>
#include "SpectraFilament.h" // Include your custom SpectralScan class

SpectralFilament spectralScanner;

void setup() {
  Serial.begin(9600); // Start serial communication at a baud rate of 9600
  while (!Serial); // Wait for the Serial Monitor to open

  Wire.begin(); // Initialize I2C communication
  if (!spectralScanner.begin()) {
    Serial.println("Sensor initialization failed!");
    while (1); // Halt if sensor initialization fails
  }
  Serial.println("Sensor initialized successfully.");
}

void loop() {
  spectralScanner.scan(); // Perform a spectral scan
  spectralScanner.printData(); // Print data to Serial Monitor
  
  delay(2000); // Wait a bit before scanning again
}
