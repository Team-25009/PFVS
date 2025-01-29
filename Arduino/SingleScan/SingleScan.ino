/* Author: Tej Scott
   Date: December 5th, 2024

   Connect SparkFun Sensor w/ Arduino
   Runs a scan w/ LED and prints upon Serial Input
*/
#include "SparkFun_AS7265X.h" // Library for AS7265X sensor
#include <Wire.h>

AS7265X sensor;
float sensorData[18];

void setup() {
  delay(10000); // Add a 5-second delay before starting Serial communication

  Serial.begin(115200);
  Serial.println("Initializing Triad Sensor Scanner...");

  // Attempt to initialize the sensor
  for (int j = 0; j < 15;) {
    if (sensor.begin() == true) {
      Serial.println("Sensor initialized successfully.");
      break;
    } else {
      Serial.print("Sensor failed to initialize, retrying... ");
      Serial.print(j + 1);
      Serial.println("/15");
      delay(1000); // Wait before retrying
      j++;
      if (j == 15) {
        Serial.println("Sensor initialization failed after 15 attempts.");
        while (1); // Halt if initialization fails
      }
    }
  }

  sensor.disableIndicator(); // Turn off the blue status LED
  Serial.println("Ready to receive scan requests...");
}

void loop() {
  // Check if there's a command from the serial input
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove any extra whitespace or newlines

    if (command == "GET_SCAN") {
      // Perform a scan with the LEDs
      sensor.takeMeasurementsWithBulb();

      // Store sensor data in the array
      sensorData[0] = sensor.getCalibratedA(); // 410nm A
      sensorData[1] = sensor.getCalibratedB(); // 435nm B
      sensorData[2] = sensor.getCalibratedC(); // 460nm C
      sensorData[3] = sensor.getCalibratedD(); // 485nm D
      sensorData[4] = sensor.getCalibratedE(); // 510nm E
      sensorData[5] = sensor.getCalibratedF(); // 535nm F
      sensorData[6] = sensor.getCalibratedG(); // 560nm G
      sensorData[7] = sensor.getCalibratedH(); // 585nm H
      sensorData[8] = sensor.getCalibratedR(); // 610nm R
      sensorData[9] = sensor.getCalibratedI(); // 645nm I
      sensorData[10] = sensor.getCalibratedS(); // 680nm S
      sensorData[11] = sensor.getCalibratedJ(); // 705nm J
      sensorData[12] = sensor.getCalibratedT(); // 730nm T
      sensorData[13] = sensor.getCalibratedU(); // 760nm U
      sensorData[14] = sensor.getCalibratedV(); // 810nm V
      sensorData[15] = sensor.getCalibratedW(); // 860nm W
      sensorData[16] = sensor.getCalibratedK(); // 900nm K
      sensorData[17] = sensor.getCalibratedL(); // 940nm L

      // Send sensor data to Serial in a single line
      for (int i = 0; i < 17; ++i) {
        Serial.print(sensorData[i]);
        Serial.print(" | ");
      }
      Serial.println(sensorData[17]); // End with the last value
    } else {
      Serial.println("Invalid command. Use 'GET_SCAN' to trigger a scan.");
    }
  }
}
