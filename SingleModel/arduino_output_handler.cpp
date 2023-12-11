/* Code adapted from The TensorFlow Authors (2020). All Rights Reserved.
==============================================================================
*/
#include "output_handler.h"

#include "Arduino.h"
#include "constants.h"

// The pin of the Arduino's built-in LED
int led = LED_BUILTIN;

// Track whether the function has run at least once
bool initialized = false;

// Animates a dot across the screen to represent the current x and y values
void HandleOutput(tflite::ErrorReporter* error_reporter, float x_value,
                  float y_value) {
  // Do this only once
  if (!initialized) {
    // Set the LED pin to output
    pinMode(led, OUTPUT);
    initialized = true;
  }

  // Calculate the brightness of the LED such that y=-1 is fully off
  // and y=1 is fully on. The LED's brightness can range from 0-255.
  int brightness = (int)(127.5f * (y_value + 1));

  // Set the brightness of the LED. If the specified pin does not support PWM,
  // this will result in the LED being on when y > 127, off otherwise.
  analogWrite(led, brightness);

  // Log the current brightness value for display in the Arduino plotter
  TF_LITE_REPORT_ERROR(error_reporter, "%d\n", brightness);
}
