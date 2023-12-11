/* Code adapted from The TensorFlow Authors (2020). All Rights Reserved.
==============================================================================
*/

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
constexpr int kTensorArenaSize = 150 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}


const int length_data = 200;  //
int vector_data[length_data] = {
  72,
  80,
  70,
  59,
  54,
  -2,
  -47,
  -72,
  -49,
  -27,
  -49,
  -77,
  -83,
  -32,
  22,
  62,
  90,
  68,
  16,
  -48,
  -47,
  -11,
  -13,
  -22,
  -10,
  21,
  -28,
  -93,
  -73,
  -3,
  65,
  63,
  17,
  -21,
  -34,
  -32,
  -33,
  2,
  45,
  43,
  39,
  47,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  72,
  80,
  70,
  59,
  54,
  -2,
  -47,
  -72,
  -49,
  -27,
  -49,
  -77,
  -83,
  -32,
  22,
  62,
  90,
  68,
  16,
  -48,
  -47,
  -11,
  -13,
  -22,
  -10,
  21,
  -28,
  -93,
  -73,
  -3,
  65,
  63,
  17,
  -21,
  -34,
  -32,
  -33,
  2,
  45,
  43,
  39,
  47,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  72,
  80,
  70,
  59,
  54,
  -2,
  -47,
  -72,
  -49,
  -27,
  -49,
  -77,
  -83,
  -32,
  22,
  62,
  90,
  68,
  16,
  -48,
  -47,
  -11,
  -13,
  -22,
  -10,
  21,
  -28,
  -93,
  -73,
  -3,
  65,
  63,
  17,
  -21,
  -34,
  -32,
  -33,
  2,
  45,
  43,
  39,
  47,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  72,
  80,
  70,
  59,
  54,
  -2,
  -47,
  -72,
  -49,
  -27,
  -49,
  -77,
  -83,
  -32,
  22,
  62,
  90,
  68,
  16,
  -48,
  -47,
  -11,
  -13,
  -22,
  -10,
  21,
  -28,
  -93,
  -73,
  -3,
  65,
  63,
  17,
  -21,
  -34,
  -32,
  -33,
  2,
  45,
  43,
  39,
  47,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
};  // dummy data


void setup() {

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter, "Model version does not match Schema version");
    return;
  }

  // This pulls in all the operation implementations we need. Actually, more than needed
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  Serial.print("Dim 1 size input: ");
  Serial.println(input->dims->data[0]);
  Serial.print("Dim 2 size inputs: ");
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

  Serial.print("Input:");
  for (int i = 0; i < length_data; ++i) {
    reinterpret_cast<float*>(input->data.raw)[i] = static_cast<float>(vector_data[i]);
    Serial.print(reinterpret_cast<float*>(input->data.raw)[i]);
    Serial.print(", ");
  }
  Serial.println("");


  // Run inference on model 1
  if (interpreter->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed for model 1");
    return;
  }

  // Process output from model 1
  for (int i = 0; i < output->dims->data[1]; ++i) {
    Serial.print("Model Output [");
    Serial.print(i);
    Serial.print("]: ");
    Serial.print(reinterpret_cast<float*>(output->data.raw)[i]);
    if (i != 2) {
      Serial.print(", ");
    }
  }
  Serial.println(" ");


  delay(2000);  // Example delay
}
