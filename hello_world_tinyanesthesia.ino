/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;
constexpr int kTensorArenaSize = 75*1024;
uint8_t tensor_arena[kTensorArenaSize];
}  

namespace {
tflite::ErrorReporter* error_reporter2 = nullptr;
const tflite::Model* model2 = nullptr;
tflite::MicroInterpreter* interpreter2 = nullptr;
TfLiteTensor* input2 = nullptr;
TfLiteTensor* output2 = nullptr;
int inference_count2 = 0;

constexpr int kTensorArenaSize2 = 65*1024;
uint8_t tensor_arena2[kTensorArenaSize2];
}  


const int length_data = 200; //
const int num_channels = 1;
int vector_data[length_data] = {
  72, 80, 70, 59, 54, -2, -47, -72, -49, -27, -49, -77, -83, -32, 22, 62, 90, 68, 16, -48, -47, -11, -13, -22, -10, 21, -28, -93, -73, -3, 65, 63, 17, -21, -34, -32, -33, 2, 45, 43, 39, 47,2,3,4,5,6,7,8,9,
  72, 80, 70, 59, 54, -2, -47, -72, -49, -27, -49, -77, -83, -32, 22, 62, 90, 68, 16, -48, -47, -11, -13, -22, -10, 21, -28, -93, -73, -3, 65, 63, 17, -21, -34, -32, -33, 2, 45, 43, 39, 47,2,3,4,5,6,7,8,9,
  
  72, 80, 70, 59, 54, -2, -47, -72, -49, -27, -49, -77, -83, -32, 22, 62, 90, 68, 16, -48, -47, -11, -13, -22, -10, 21, -28, -93, -73, -3, 65, 63, 17, -21, -34, -32, -33, 2, 45, 43, 39, 47,2,3,4,5,6,7,8,9,
  72, 80, 70, 59, 54, -2, -47, -72, -49, -27, -49, -77, -83, -32, 22, 62, 90, 68, 16, -48, -47, -11, -13, -22, -10, 21, -28, -93, -73, -3, 65, 63, 17, -21, -34, -32, -33, 2, 45, 43, 39, 47,2,3,4,5,6,7,8,9,
};


void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  static tflite::MicroErrorReporter micro_error_reporter2;
  error_reporter2 = &micro_error_reporter2;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  model2 = tflite::GetModel(g_model2);

  if (model->version() != TFLITE_SCHEMA_VERSION || model2->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model version does not match Schema version");
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  static tflite::MicroInterpreter static_interpreter2(
      model2, resolver, tensor_arena2, kTensorArenaSize2, error_reporter2);
  interpreter2 = &static_interpreter2;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  
  TfLiteStatus allocate_status2 = interpreter2->AllocateTensors();
  if (allocate_status2 != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "for the second model AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  input2 = interpreter2->input(0);
  output2 = interpreter2->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
  inference_count2 = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  Serial.print("Dim 1 size input: ");
  Serial.println(input->dims->data[0]);
  Serial.print("Dim 2 size inputs: ");
  Serial.println(input->dims->data[1]);
  delay(1000);
  Serial.print("Number of dimensions output: ");
  Serial.print("Dim 1 size output: ");
  Serial.println(output->dims->data[0]);
  Serial.print("Dim 2 size output: ");
  Serial.println(output->dims->data[1]);

  Serial.println("#######################");
  Serial.println("this is for second model :D");
  Serial.print("Dim 1 size input: ");
  Serial.println(input2->dims->data[0]);
  Serial.print("Dim 2 size inputs: ");
  Serial.println(input2->dims->data[1]);
  delay(1000);
  Serial.print("Number of dimensions output: ");
  Serial.print("Dim 1 size output: ");
  Serial.println(output2->dims->data[0]);
  Serial.print("Dim 2 size output: ");
  Serial.println(output2->dims->data[1]);
  Serial.println("#######################");
  Serial.println("starts the input");

  for (int i = 0; i < 200; ++i) {
    input->data.int8[i] = vector_data[i];
    input2->data.int8[i] = vector_data[i];
    Serial.print(input2->data.int8[i]);
    Serial.print(", ");
  }
  Serial.println("this was the input");
  

  // Run inference on model 1
  if (interpreter->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed for model 1");
    return;
  }

  // Process output from model 1
  for (int i = 0; i < output->dims->data[1]; ++i) {
    // Example: Print output value
    Serial.print("Model 1 Output [");
    Serial.print(i);
    Serial.print("]: ");
    Serial.println(output->data.int8[i]);
  }

  // Run inference on model 2
  if (interpreter2->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed for model 2");
    return;
  }

  // Process output from model 2
  for (int i = 0; i < output2->dims->data[1]; ++i) {
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
