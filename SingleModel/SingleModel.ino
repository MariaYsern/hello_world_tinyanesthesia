/* Code adapted from The TensorFlow Authors (2020). All Rights Reserved.
==============================================================================
*/

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model.h"
#include "input_vector.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;
constexpr int kTensorArenaSize = 150 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
const int window_length = 200;
const int num_windows = input_vector_length / window_length;
}


void setup() {
  // static tflite::MicroErrorReporter micro_error_reporter;
  // error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need. Actually, more than needed
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  Serial.println("-----------START-----------");
  Serial.print("Dim 1 size input: ");
  Serial.println(input->dims->data[0]);
  Serial.print("Dim 2 size input: ");
  Serial.println(input->dims->data[1]);
  Serial.print("Dim 1 size output: ");
  Serial.println(output->dims->data[0]);
  Serial.print("Dim 2 size output: ");
  Serial.println(output->dims->data[1]);

  // Your output should be the following:
  // Dim 1 size input: 1
  // Dim 2 size inputs: 200
  // Dim 1 size output: 1
  // Dim 2 size output: 3
  float accuracy = 0;
  for (int i = 0; i < num_windows; ++i) {
    int start_index = i * window_length;
    int end_index = (i + 1) * window_length;
    int correct_label = labels_vector[i];
    
    Serial.print("Sample ");
    Serial.print(i);
    Serial.print(" of ");
    Serial.print(num_windows);
    Serial.print(". Correct label: ");
    Serial.print(correct_label);
    
    // Insert windowed input to input tensor
    for (int j = start_index; j < end_index; ++j) {
      reinterpret_cast<float*>(input->data.raw)[j] = static_cast<float>(input_vector_data[j]);
    }

    // Run inference on model 1
    if (interpreter->Invoke() != kTfLiteOk) {
      MicroPrintf("Invoke failed for model 1");
      return;
    }

    // Process output from model 1
    Serial.print(". Predicted label: (");
    float max_value = 0;
    int max_position = 0;
    for (int i = 0; i < output->dims->data[1]; ++i) {
      Serial.print(reinterpret_cast<float*>(output->data.raw)[i]);
      if (i < output->dims->data[1] - 1) {
        Serial.print(", ");
      } else {
        Serial.print(") = ");
      }
      
      if (reinterpret_cast<float*>(output->data.raw)[i] > max_value) {
        max_value = reinterpret_cast<float*>(output->data.raw)[i];
        max_position = i;
      }
    }
    if (max_position == correct_label) {
      accuracy = accuracy + 1;
    }
    Serial.println(max_position);
  }
  float acc_score = accuracy / num_windows;
  Serial.print("Accuracy score = ");
  Serial.print(accuracy);
  Serial.print("/");
  Serial.print(num_windows);
  Serial.print("=");
  Serial.println(acc_score);
  Serial.println("-----------END-----------");

  delay(120000);  // Example delay
}
