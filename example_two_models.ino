#include <TensorFlowLite.h>

#include "main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model1.h" // Model 1
#include "model2.h" // Model 2
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
tflite::ErrorReporter* error_reporter = nullptr;

// Models
const tflite::Model* model1 = nullptr;
const tflite::Model* model2 = nullptr;

// Interpreters
tflite::MicroInterpreter* interpreter1 = nullptr;
tflite::MicroInterpreter* interpreter2 = nullptr;

// Tensors
TfLiteTensor* input1 = nullptr;
TfLiteTensor* output1 = nullptr;
TfLiteTensor* input2 = nullptr;
TfLiteTensor* output2 = nullptr;

// Tensor Arenas
constexpr int kTensorArenaSize1 = 80*1024; // Adjust size as needed
constexpr int kTensorArenaSize2 = 80*1024; // Adjust size as needed
uint8_t tensor_arena1[kTensorArenaSize1];
uint8_t tensor_arena2[kTensorArenaSize2];

// The name of this function is important for Arduino compatibility.
void setup() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load both models
  model1 = tflite::GetModel(model1_data);
  model2 = tflite::GetModel(model2_data);

  // Verify model version
  if (model1->version() != TFLITE_SCHEMA_VERSION || model2->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model version does not match Schema version");
    return;
  }

  static tflite::AllOpsResolver resolver;

  // Create interpreters for each model
  static tflite::MicroInterpreter static_interpreter1(
      model1, resolver, tensor_arena1, kTensorArenaSize1, error_reporter);
  interpreter1 = &static_interpreter1;

  static tflite::MicroInterpreter static_interpreter2(
      model2, resolver, tensor_arena2, kTensorArenaSize2, error_reporter);
  interpreter2 = &static_interpreter2;

  // Allocate tensors
  if (interpreter1->AllocateTensors() != kTfLiteOk || interpreter2->AllocateTensors() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors
  input1 = interpreter1->input(0);
  output1 = interpreter1->output(0);
  input2 = interpreter2->input(0);
  output2 = interpreter2->output(0);
}

void loop() {
  // Prepare input data for model 1
  for (size_t i = 0; i < input1->bytes; ++i) {
    input1->data.int8[i] = ... // Assign input data for model 1
  }

  // Run inference on model 1
  if (interpreter1->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed for model 1");
    return;
  }

  // Process output from model 1
  for (size_t i = 0; i < output1->bytes; ++i) {
    // Example: Print output value
    Serial.print("Model 1 Output [");
    Serial.print(i);
    Serial.print("]: ");
    Serial.println(output1->data.int8[i]);
  }

  // Prepare input data for model 2
  for (size_t i = 0; i < input2->bytes; ++i) {
    input2->data.int8[i] = ... // Assign input data for model 2
  }

  // Run inference on model 2
  if (interpreter2->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed for model 2");
    return;
  }

  // Process output from model 2
  for (size_t i = 0; i < output2->bytes; ++i) {
    // Example:
    // Example: Print output value
    Serial.print("Model 2 Output [");
    Serial.print(i);
    Serial.print("]: ");
    Serial.println(output2->data.int8[i]);
  }

  // Add any additional logic or delays as needed
  delay(1000); // Example delay
}
