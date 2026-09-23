/*
 * Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.litert.resizesmoke;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.widget.TextView;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.LiteRtException;
import com.google.ai.edge.litert.TensorBuffer;
import java.util.Arrays;
import java.util.Collections;

/** A minimal on-device smoke test for CompiledModel input resizing. */
public final class CompiledModelResizeSmokeTestActivity extends Activity {
  private static final String TAG = "LiteRtResizeSmoke";
  private static final String DYNAMIC_MODEL_ASSET = "dynamic_add.tflite";
  private static final String STATIC_MODEL_ASSET = "simple_add.tflite";
  private static final String INPUT_0 = "arg0";
  private static final String INPUT_1 = "arg1";
  private static final String OUTPUT = "tfl.add";

  private TextView status;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    status = new TextView(this);
    status.setGravity(Gravity.CENTER);
    status.setPadding(48, 48, 48, 48);
    status.setText("Running LiteRT resize smoke test...");
    setContentView(status);

    new Thread(this::runAndReport, "litert-resize-smoke").start();
  }

  private void runAndReport() {
    try {
      String result = runSmokeTest();
      Log.i(TAG, result);
      runOnUiThread(() -> status.setText(result));
    } catch (Throwable error) {
      String result = "FAIL: " + error.getClass().getSimpleName() + ": " + error.getMessage();
      Log.e(TAG, result, error);
      runOnUiThread(() -> status.setText(result));
    }
  }

  private String runSmokeTest() throws LiteRtException {
    runStrictResizeSmokeTest();
    runNonStrictResizeSmokeTest();
    return "PASS: CompiledModel strict and non-strict input resizing succeeded.";
  }

  private void runStrictResizeSmokeTest() throws LiteRtException {
    int[] dimensions = {2, 2, 3};
    float[] input0Values = new float[12];
    float[] input1Values = new float[12];
    Arrays.fill(input0Values, 1.0f);
    Arrays.fill(input1Values, 2.0f);

    try (CompiledModel model = CompiledModel.create(getAssets(), DYNAMIC_MODEL_ASSET)) {
      model.resizeInputTensor(INPUT_0, dimensions);
      model.resizeInputTensor(INPUT_1, dimensions);
      assertBufferSize(model.getInputBufferRequirements(INPUT_0, "").getBufferSize(), 48);
      assertBufferSize(model.getInputBufferRequirements(INPUT_1, "").getBufferSize(), 48);

      try (TensorBuffer input0 = model.createInputBuffer(INPUT_0, "");
          TensorBuffer input1 = model.createInputBuffer(INPUT_1, "");
          TensorBuffer output = model.createOutputBuffer(OUTPUT, "")) {
        input0.writeFloat(input0Values);
        input1.writeFloat(input1Values);
        model.run(Arrays.asList(input0, input1), Collections.singletonList(output), 0);
        assertFloatArray(output.readFloat(), fill(12, 3.0f));
      }
    }
  }

  private void runNonStrictResizeSmokeTest() throws LiteRtException {
    int[] dimensions = {3};

    try (CompiledModel model = CompiledModel.create(getAssets(), STATIC_MODEL_ASSET)) {
      expectLiteRtException(() -> model.resizeInputTensor(INPUT_0, dimensions));
      model.resizeInputTensorNonStrict(INPUT_0, dimensions);
      model.resizeInputTensorNonStrict(INPUT_1, dimensions);
      assertBufferSize(model.getInputBufferRequirements(INPUT_0, "").getBufferSize(), 12);
      assertBufferSize(model.getInputBufferRequirements(INPUT_1, "").getBufferSize(), 12);

      try (TensorBuffer input0 = model.createInputBuffer(INPUT_0, "");
          TensorBuffer input1 = model.createInputBuffer(INPUT_1, "");
          TensorBuffer output = model.createOutputBuffer(OUTPUT, "")) {
        input0.writeFloat(new float[] {1.0f, 2.0f, 3.0f});
        input1.writeFloat(new float[] {10.0f, 20.0f, 30.0f});
        model.run(Arrays.asList(input0, input1), Collections.singletonList(output), 0);
        assertFloatArray(output.readFloat(), new float[] {11.0f, 22.0f, 33.0f});
      }
    }
  }

  private static float[] fill(int size, float value) {
    float[] values = new float[size];
    Arrays.fill(values, value);
    return values;
  }

  private static void assertBufferSize(int actual, int expected) {
    if (actual != expected) {
      throw new AssertionError("Expected buffer size " + expected + ", got " + actual + ".");
    }
  }

  private static void assertFloatArray(float[] actual, float[] expected) {
    if (!Arrays.equals(actual, expected)) {
      throw new AssertionError("Unexpected inference output: " + Arrays.toString(actual));
    }
  }

  private static void expectLiteRtException(ThrowingRunnable runnable) {
    try {
      runnable.run();
      throw new AssertionError("Expected LiteRtException.");
    } catch (LiteRtException expected) {
      // Expected.
    }
  }

  private interface ThrowingRunnable {
    void run() throws LiteRtException;
  }
}
