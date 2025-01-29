#ifndef SPECTRALFILAMENT_H
#define SPECTRALFILAMENT_H

#include <Wire.h>
#include <SparkFun_AS7265X.h>

class SpectralFilament {
  private:
    AS7265X sensor; // Create an instance of the AS7265X class
    float wavelengths[18];
    uint16_t intensities[18];

  public:
    SpectralFilament() {
      // Set default wavelengths for the AS7265x sensor bands
      wavelengths[0] = 410;
      wavelengths[1] = 435;
      wavelengths[2] = 460;
      wavelengths[3] = 485;
      wavelengths[4] = 510;
      wavelengths[5] = 535;
      wavelengths[6] = 560;
      wavelengths[7] = 585;
      wavelengths[8] = 610;
      wavelengths[9] = 645;
      wavelengths[10] = 680;
      wavelengths[11] = 705;
      wavelengths[12] = 730;
      wavelengths[13] = 760;
      wavelengths[14] = 810;
      wavelengths[15] = 860;
      wavelengths[16] = 900;
      wavelengths[17] = 940;
    }

    bool begin() {
      return sensor.begin();
    }

    // Perform a scan and populate intensity data
    void scan() {
      sensor.takeMeasurements();
      intensities[0] = sensor.getCalibratedA(); // 410nm
      intensities[1] = sensor.getCalibratedB(); // 435nm
      intensities[2] = sensor.getCalibratedC(); // 460nm
      intensities[3] = sensor.getCalibratedD(); // 485nm
      intensities[4] = sensor.getCalibratedE(); // 510nm
      intensities[5] = sensor.getCalibratedF(); // 535nm
      intensities[6] = sensor.getCalibratedG(); // 560nm
      intensities[7] = sensor.getCalibratedH(); // 585nm
      intensities[8] = sensor.getCalibratedR(); // 610nm
      intensities[9] = sensor.getCalibratedI(); // 645nm
      intensities[10] = sensor.getCalibratedS(); // 680nm
      intensities[11] = sensor.getCalibratedJ(); // 705nm
      intensities[12] = sensor.getCalibratedT(); // 730nm
      intensities[13] = sensor.getCalibratedU(); // 760nm
      intensities[14] = sensor.getCalibratedV(); // 810nm
      intensities[15] = sensor.getCalibratedW(); // 860nm
      intensities[16] = sensor.getCalibratedK(); // 900nm
      intensities[17] = sensor.getCalibratedL(); // 940nm
    }

    // Print data to the serial monitor
    void printData() {
      Serial.println("Wavelength\tIntensity");
      for (int i = 0; i < 18; i++) {
        Serial.print(wavelengths[i]);
        Serial.print("\t\t");
        Serial.println(intensities[i]);
      }
    }

    // Return the wavelength with the highest intensity
    float getPeakWavelength() {
      uint16_t peakIntensity = 0;
      float peakWavelength = wavelengths[0];
      for (int i = 0; i < 18; i++) {
        if (intensities[i] > peakIntensity) {
          peakIntensity = intensities[i];
          peakWavelength = wavelengths[i];
        }
      }
      return peakWavelength;
    }

    // Get the intensity of a specific wavelength
    uint16_t getIntensity(float wavelength) {
      for (int i = 0; i < 18; i++) {
        if (wavelengths[i] == wavelength) {
          return intensities[i];
        }
      }
      return 0; // Return 0 if wavelength not found
    }

    // Get data for plotting
    void getGraphData(float *wavelengthsOut, uint16_t *intensitiesOut) {
      scan();  // Perform a scan
      for (int i = 0; i < 18; i++) {
        wavelengthsOut[i] = wavelengths[i];
        intensitiesOut[i] = intensities[i];
      }
    }
};

#endif