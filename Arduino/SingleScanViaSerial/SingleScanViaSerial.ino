/* Author: Tej Scott
  Date: November 11th, 2024
  
  Connect SparkFun Sensor w/ Arduino
  Runs a scan w/ LED and prints upon Serial Input
  */
#include "SparkFun_AS7265X.h" //Click here to get the library: http://librarymanager/All#SparkFun_AS7265X
AS7265X sensor;

#include <Wire.h>

  // float dataA, dataB, dataC, dataD, dataE, dataF; //First 6 float storage
  // float dataG, dataH, dataR, dataI, dataS, dataJ; //Middle 6 float storage
  // float dataT, dataU, dataV, dataW, dataK, dataL; //Last 6 float storage
  float sensorData[18];

void setup() {
  
  Serial.begin(115200);
  Serial.println("Triad Sensor Scanner Example");

  for (int j=0; j<15;) {
      sensor.begin();
      delay (50);

      if (sensor.begin()==true)
      j=50;

      else {
        Serial.print("Sensor failed to initialize, retrying... ");
        Serial.print(j+1);  
        Serial.println("/15");

        j++;
      }
  }

  sensor.disableIndicator(); //Turn off the blue status LED

  Serial.println("A,B,C,D,E,F,G,H,R,I,S,J,T,U,V,W,K,L");
}

void loop() {
  Serial.println("Point the Triad away and press a key to initiate scan...");
  while (Serial.available() == false) {} //Do nothing while we wait for user to press a key
  Serial.read(); //Throw away the user's button
  Serial.flush();

  sensor.takeMeasurementsWithBulb(); //Get scan with LEDs

  //Store sensor data to variables
  sensorData[0] = sensor.getCalibratedA(); //410nm A
  sensorData[1] = sensor.getCalibratedB(); //435nm B
  sensorData[2] = sensor.getCalibratedC(); //460nm C
  sensorData[3] = sensor.getCalibratedD(); //485nm D
  sensorData[4] = sensor.getCalibratedE(); //510nm E
  sensorData[5] = sensor.getCalibratedF(); //535nm F

  sensorData[6] = sensor.getCalibratedG(); //560nm G
  sensorData[7] = sensor.getCalibratedH(); //585nm H
  sensorData[8] = sensor.getCalibratedR(); //610nm R
  sensorData[9] = sensor.getCalibratedI(); //645nm I
  sensorData[10] = sensor.getCalibratedS(); //680nm S
  sensorData[11] = sensor.getCalibratedJ(); //705nm J

  sensorData[12] = sensor.getCalibratedT(); //730nm T 
  sensorData[13] = sensor.getCalibratedU(); //760nm U
  sensorData[14] = sensor.getCalibratedV(); //810nm V
  sensorData[15] = sensor.getCalibratedW(); //860nm W
  sensorData[16] = sensor.getCalibratedK(); //900nm K
  sensorData[17] = sensor.getCalibratedL(); //940nm L

  //Print data from variables
  for(int i = 0; i < 17; ++i) {
    Serial.print(sensorData[i]);
    Serial.print(" | ");
  }
  Serial.print(sensorData[17]);
  Serial.println();
}
