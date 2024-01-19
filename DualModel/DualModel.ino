/* Code adapted from The TensorFlow Authors (2020). All Rights Reserved.
==============================================================================
*/

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model.h"
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
constexpr int kTensorArenaSize = 175 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}

namespace {
tflite::ErrorReporter* error_reporter2 = nullptr;
const tflite::Model* model2 = nullptr;
tflite::MicroInterpreter* interpreter2 = nullptr;
TfLiteTensor* input2 = nullptr;
TfLiteTensor* output2 = nullptr;
int inference_count2 = 0;

constexpr int kTensorArenaSize2 = 2 * 1024;
uint8_t tensor_arena2[kTensorArenaSize2];
}


const int length_data = 200; //
// dummy data (arranged to be categorized as 'sleep' for application in the second model). Also note that is stored as int (not float)
float vector_data[200] = {0.2902152770704095, 0.3593359143431327, 0.4828908807540264, 0.5908069878033232, 0.5875870192470262, 0.6274629006193031, 0.607009652430606, 0.5399528374963548, 0.5225671838285278, 0.5658517425984066, 0.6174971872706353, 0.668792483250771, 0.7560725242845391, 0.8387598822604063, 0.840354857159972, 0.8918947124873685, 0.9333574747689901, 0.9489804522743339, 0.9842774437532347, 1.0341825662265802, 1.1076798572520135, 1.1820992264940349, 1.2433711318003533, 1.253355112380536, 1.249254870933928, 1.2379040621904545, 1.2911287430194147, 1.4431772925666335, 1.5686202672001395, 1.6206000223454218, 1.5911098970188808, 1.5507257977294486, 1.5149614640586822, 1.50042495745645, 1.4998696126142543, 1.4628589478621918, 1.4197791810617446, 1.3706269308218888, 1.374942617884226, 1.3785927380495726, 1.4047597658543787, 1.432704658211137, 1.4445290780840383, 1.436207680800487, 1.3836170309632543, 1.295299081296777, 1.3007464761045495, 1.2452829209469927, 1.1528333220363605, 1.0887907184745589, 1.0752156874047336, 1.0535674901494239, 0.9796142566896576, 0.8790922152768555, 0.7968808811035784, 0.7583330298938237, 0.7356078188619716, 0.7304237639405621, 0.7492704767784439, 0.644503967631429, 0.6245776224344353, 0.592140913779254, 0.5377200390343235, 0.5358103351453475, 0.47996280036345446, 0.4721832814438693, 0.4376760820271747, 0.46338150129380484, 0.46268446554641907, 0.44473439629899786, 0.41238937516029245, 0.3066011752207631, 0.20425795311701148, 0.13438545634017052, 0.12295384463903082, 0.1971185591105209, 0.2376724302378438, 0.24813950800638782, 0.25492985084283815, 0.26256131361260854, 0.2607016853448478, 0.3007380619720297, 0.34045087551076864, 0.3841790611495736, 0.4245634984918682, 0.4010283405792197, 0.40245884299494233, 0.40379046110111244, 0.44502210417484717, 0.4887556330150023, 0.5231063795146598, 0.5551126966837421, 0.6126112823673275, 0.5571652964244377, 0.5803794769893469, 0.5668690112011978, 0.5461278570996994, 0.44466003826606454, 0.4067082875042247, 0.37441001035607613, 0.37731582168279537, 0.3908674223414044, 0.38461723502547174, 0.40022785311657016, 0.4296142112931722, 0.43806163543459115, 0.44937924946710023, 0.47298605410564865, 0.5656288642754606, 0.6355339494607781, 0.6774437758065601, 0.690972702616641, 0.6673640939969799, 0.7080155063811122, 0.7152668554576304, 0.7365129431777666, 0.7899823364975253, 0.8464608995160927, 0.8267087602625091, 0.8274202861128697, 0.8496602520441727, 0.8271055825134375, 0.8290107964668723, 0.8567407663050941, 0.8363584283225255, 0.7623503026767087, 0.7002007010075859, 0.6370768603917915, 0.5834372399254513, 0.5408875444251441, 0.47780106113952336, 0.46584324404727095, 0.43742518828354476, 0.3872877230337237, 0.293031996329455, 0.29126064330278817, 0.27829363894790204, 0.20742856666889126, 0.15276163862438752, 0.08844934732647076, 0.06564264811207848, -0.01462659886530084, -0.07360280129336352, -0.11202800396716527, -0.14994489450126128, -0.19344248273674178, -0.27334558823409855, -0.3771659158065305, -0.4243360868905694, -0.47065065284549157, -0.5111622793156271, -0.5566086269280781, -0.6181569907632888, -0.6908600212548446, -0.7916961884734415, -0.8600852557508094, -0.8532548062941078, -0.8505064167199068, -0.9022440412682162, -0.9593299133393619, -0.9608982849472515, -0.9775595195501657, -1.002338389573595, -1.0099286322044434, -1.0324364395201784, -1.0950144416897394, -1.146545300600212, -1.2001643577532188, -1.2081754190161278, -1.2532288778665615, -1.2595313466501825, -1.2558141136059913, -1.2552137630836266, -1.2297974414608284, -1.2425994519132049, -1.2415821026933842, -1.2132464818934048, -1.1670292204339112, -1.1722569792417226, -1.2034083070143116, -1.248820143926641, -1.2983328781343324, -1.2877760893142964, -1.284586044848209, -1.3099780483923074, -1.3018558561144278, -1.308937336858362, -1.3802997164378181, -1.3918024143816006, -1.401541841354332, -1.3867874087315635, -1.3917651580945238, -1.4016018782934734, -1.4293410871189474, -1.4120036000573764, -1.4048333382361398, -1.4079194155842392, -1.375370951419658, -1.3690336761000441, -1.3577822126673966};

void setup() {

  model = tflite::GetModel(g_model);
  model2 = tflite::GetModel(g_model2);

  if (model->version() != TFLITE_SCHEMA_VERSION || model2->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Models provided are schema version %d and %d not equal "
        "to supported version %d.",
        model->version(), model2->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  static tflite::MicroInterpreter static_interpreter2(
    model2, resolver, tensor_arena2, kTensorArenaSize2);
  interpreter2 = &static_interpreter2;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed in first model");
    return;
  }

  TfLiteStatus allocate_status2 = interpreter2->AllocateTensors();
  if (allocate_status2 != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed in second model");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  input2 = interpreter2->input(0);
  output2 = interpreter2->output(0);
}

void loop() {
  Serial.println("-----------START-----------");
  Serial.print("Dim 1 size input: ");
  Serial.println(input->dims->data[0]);
  Serial.print("Dim 2 size inputs: ");
  Serial.println(input->dims->data[1]);
  Serial.print("Dim 1 size output: ");
  Serial.println(output->dims->data[0]);
  Serial.print("Dim 2 size output: ");
  Serial.println(output->dims->data[1]);
  delay(1000);

  Serial.println("-----------------------");
  Serial.println("this is for second model:");
  Serial.print("Dim 1 size input: ");
  Serial.println(input2->dims->data[0]);
  Serial.print("Dim 2 size inputs: ");
  Serial.println(input2->dims->data[1]);
  Serial.print("Dim 1 size output: ");
  Serial.println(output2->dims->data[0]);
  Serial.print("Dim 2 size output: ");
  Serial.println(output2->dims->data[1]);
  Serial.println("-----------------------");
  Serial.println("starts the input");

  Serial.print("Input:");
  for (int i = 0; i < length_data; ++i) {
    reinterpret_cast<float*>(input->data.raw)[i] = static_cast<float>(vector_data[i]);
    reinterpret_cast<float*>(input2->data.raw)[i] = static_cast<float>(vector_data[i]); // the input is the same for both models
    Serial.print(reinterpret_cast<float*>(input->data.raw)[i]);
    Serial.print(", ");
  }
  Serial.println("");

  // Run inference on model 1
  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed for model 1");
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

  // Check if output[1] is greater than output[0] and run the second model if true
  if (reinterpret_cast<float*>(output->data.raw)[1] > reinterpret_cast<float*>(output->data.raw)[0]) {
    // Run inference on model 2
    if (interpreter2->Invoke() != kTfLiteOk) {
      MicroPrintf("Invoke failed for model 2");
      return;
    }

    // Process output from model 2
    Serial.println("Model 2 Output:");
    for (int i = 0; i < output2->dims->data[1]; ++i) {
      Serial.print("Output2 [");
      Serial.print(i);
      Serial.print("]: ");
      Serial.print(reinterpret_cast<float*>(output2->data.raw)[i]);
      if (i != output2->dims->data[1] - 1) {
        Serial.print(", ");
      }
    }
    Serial.println(" ");
    Serial.println("-----------END-----------");
    delay(60000);  // Example delay
  }
}

