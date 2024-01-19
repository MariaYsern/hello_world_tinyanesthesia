/* Code adapted from The TensorFlow Authors (2020). All Rights Reserved.
==============================================================================
*/
#include "output_handler.h"

#include "Arduino.h"
#include "constants.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_log.h"

// The pin of the Arduino's built-in LED
int led = LED_BUILTIN;

// Track whether the function has run at least once
bool initialized = false;

// Animates a dot across the screen to represent the current x and y values
void HandleOutput(float x_value, float y_value) {
  // Do this only once
  if (!initialized) {
    // Set the LED pin to output
    pinMode(led, OUTPUT);
    initialized = true;
  }

  // Calculate the brightness of the LED such that y=-1 is fully off
  // and y=1 is fully on. The LED's brightness can range from 0-255.
  int brightness = (int)(127.5f * (y_value + 1));

  // The y value is not actually constrained to the range [-1, 1], so we need to
  // clamp the brightness value before sending it to the PWM/LED.
  int brightness_clamped = std::min(255, std::max(0, brightness));

  // Set the brightness of the LED. If the specified pin does not support PWM,
  // this will result in the LED being on when y > 127, off otherwise.
  analogWrite(led, brightness);

  // Log the current brightness value for display in the Arduino plotter
  MicroPrintf("%d\n", brightness);
  delay(33);
}
