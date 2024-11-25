/*
  Read the 18 channels of spectral light over I2C using the Spectral Triad
  By: Nathan Seidle
  SparkFun Electronics
  Date: October 25th, 2018
  License: MIT. See license file for more information but you can
  basically do whatever you want with this code.

  This example takes all 18 readings and blinks the illumination LEDs 
  as it goes. We recommend you point the Triad away from your eyes, the LEDs are *bright*.
  
  Feel like supporting open source hardware?
  Buy a board from SparkFun! https://www.sparkfun.com/products/15050

  Hardware Connections:
  Plug a Qwiic cable into the Spectral Triad and a BlackBoard
  If you don't have a platform with a Qwiic connection use the SparkFun Qwiic Breadboard Jumper (https://www.sparkfun.com/products/14425)
  Open the serial monitor at 115200 baud to see the output
*/

#include "SparkFun_AS7265X.h" //Click here to get the library: http://librarymanager/All#SparkFun_AS7265X
AS7265X sensor;

#include <Wire.h>

void setup()
{
  Serial.begin(115200);
  Serial.println("AS7265x Spectral Triad Example");

  Serial.println("Point the Triad away and press a key to begin with illumination...");
  while (Serial.available() == false)
  {
  }              //Do nothing while we wait for user to press a key
  Serial.read(); //Throw away the user's button
  Serial.flush();

  for (int j=0; j<25;) {
      sensor.begin();
      delay (50);

      if (sensor.begin()==true)
      j=50;

      else {
        Serial.print("Sensor failed to initialize, retrying... ");
        Serial.print(j+1);  
        Serial.println("/25");

        j++;
      }
    }

  sensor.disableIndicator(); //Turn off the blue status LED

  //Experimentation Code Here
  sensor.setGain(AS7265X_GAIN_64X); //1 37 16 64(Default?)

  //sensor.setMeasurementMode(AS7265X_MEASUREMENT_MODE_4CHAN); //Channels STUV on x51
  //sensor.setMeasurementMode(AS7265X_MEASUREMENT_MODE_4CHAN_2); //Channels RTUW on x51
  //sensor.setMeasurementMode(AS7265X_MEASUREMENT_MODE_6CHAN_CONTINUOUS); //All 6 channels on all devices
  //sensor.setMeasurementMode(AS7265X_MEASUREMENT_MODE_6CHAN_ONE_SHOT); //Default: All 6 channels, all devices, just once

  //sensor.setIntegrationCycles(49); //Default 50*2.8ms = 140ms per reading
  //sensor.setIntegrationCycles(1); //2*2.8ms = 5.6ms per reading // ~~~~~Doesn't work~~~~~~
  
  //White LED
  //sensor.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_12_5MA, AS7265x_LED_WHITE); //Default
  //sensor.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_25MA, AS7265x_LED_WHITE); //Allowed
  //sensor.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_50MA, AS7265x_LED_WHITE); //Allowed
  sensor.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_100MA, AS7265x_LED_WHITE); //Allowed


  //IR LED has max forward current of 65mA
  //sensor.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_12_5MA, AS7265x_LED_IR); //Default
  //sensor.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_25MA, AS7265x_LED_IR); //Allowed
  sensor.setBulbCurrent(AS7265X_LED_CURRENT_LIMIT_50MA, AS7265x_LED_IR); //Allowed
  

  Serial.println("A,B,C,D,E,F,G,H,R,I,S,J,T,U,V,W,K,L");
}

void loop()
{
  for(int i = 0; i < 100; i++) {
    sensor.takeMeasurementsWithBulb(); //Light Measurements
    //sensor.takeMeasurements(); //Dark Measurements

//CALIBRATED
  Serial.print(sensor.getCalibratedA()); //410nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedB()); //435nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedC()); //460nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedD()); //485nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedE()); //510nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedF()); //535nm
  Serial.print(",");

  Serial.print(sensor.getCalibratedG()); //560nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedH()); //585nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedR()); //610nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedI()); //645nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedS()); //680nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedJ()); //705nm
  Serial.print(",");

  Serial.print(sensor.getCalibratedT()); //730nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedU()); //760nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedV()); //810nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedW()); //860nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedK()); //900nm
  Serial.print(",");
  Serial.print(sensor.getCalibratedL()); //940nm
  
  Serial.println();
//RAW
  Serial.print(sensor.getA()); //410nm
  Serial.print(",");
  Serial.print(sensor.getB()); //435nm
  Serial.print(",");
  Serial.print(sensor.getC()); //460nm
  Serial.print(",");
  Serial.print(sensor.getD()); //485nm
  Serial.print(",");
  Serial.print(sensor.getE()); //510nm
  Serial.print(",");
  Serial.print(sensor.getF()); //535nm
  Serial.print(",");

  Serial.print(sensor.getG()); //560nm
  Serial.print(",");
  Serial.print(sensor.getH()); //585nm
  Serial.print(",");
  Serial.print(sensor.getR()); //610nm
  Serial.print(",");
  Serial.print(sensor.getI()); //645nm
  Serial.print(",");
  Serial.print(sensor.getS()); //680nm
  Serial.print(",");
  Serial.print(sensor.getJ()); //705nm
  Serial.print(",");

  Serial.print(sensor.getT()); //730nm
  Serial.print(",");
  Serial.print(sensor.getU()); //760nm
  Serial.print(",");
  Serial.print(sensor.getV()); //810nm
  Serial.print(",");
  Serial.print(sensor.getW()); //860nm
  Serial.print(",");
  Serial.print(sensor.getK()); //900nm
  Serial.print(",");
  Serial.print(sensor.getL()); //940nm

  Serial.println();
  }
  Serial.println("100 Samples Scanned\n\n\n");
  return;
}
