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

package com.google.ai.edge.litert.int16smoke;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.widget.TextView;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.LiteRtException;
import com.google.ai.edge.litert.TensorBuffer;
import com.google.ai.edge.litert.TensorType;
import java.util.Arrays;
import java.util.Collections;

/** A minimal on-device smoke test for INT16 I/O through the CompiledModel API. */
public final class Int16SmokeTestActivity extends Activity {
  private static final String TAG = "LiteRtInt16Smoke";
  private static final String MODEL_ASSET = "int16_add.tflite";
  private static final String INPUT_1 = "input_1";
  private static final String INPUT_2 = "input_2";
  private static final String OUTPUT = "add";
  private static final int ELEMENT_COUNT = 32 * 32;

  private TextView status;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    status = new TextView(this);
    status.setGravity(Gravity.CENTER);
    status.setPadding(48, 48, 48, 48);
    status.setText("Running LiteRT INT16 smoke test...");
    setContentView(status);

    new Thread(this::runAndReport, "litert-int16-smoke").start();
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
    short[] signedValues = new short[ELEMENT_COUNT];
    signedValues[0] = Short.MIN_VALUE;
    signedValues[1] = -123;
    signedValues[2] = 0;
    signedValues[3] = Short.MAX_VALUE;
    for (int i = 4; i < ELEMENT_COUNT; i++) {
      signedValues[i] = (short) (i - 512);
    }

    try (CompiledModel model = CompiledModel.create(getAssets(), MODEL_ASSET);
        TensorBuffer input1 = model.createInputBuffer(INPUT_1, "");
        TensorBuffer input2 = model.createInputBuffer(INPUT_2, "");
        TensorBuffer output = model.createOutputBuffer(OUTPUT, "")) {
      assertInt16(model.getInputTensorType(INPUT_1, ""), INPUT_1);
      assertInt16(model.getInputTensorType(INPUT_2, ""), INPUT_2);
      assertInt16(model.getOutputTensorType(OUTPUT, ""), OUTPUT);

      try {
        input1.writeInt16(new short[ELEMENT_COUNT + 1]);
        throw new AssertionError("Oversize INT16 write did not fail.");
      } catch (LiteRtException expected) {
        // Expected: the tensor only holds ELEMENT_COUNT elements.
      }

      input1.writeInt16(signedValues);
      if (!Arrays.equals(signedValues, input1.readInt16())) {
        throw new AssertionError("INT16 input round-trip changed signed values.");
      }

      input2.writeInt16(new short[ELEMENT_COUNT]);
      model.run(Arrays.asList(input1, input2), Collections.singletonList(output), 0);

      short[] outputValues = output.readInt16();
      if (outputValues.length != ELEMENT_COUNT
          || outputValues[0] >= 0
          || outputValues[1] >= 0
          || outputValues[2] != 0
          || outputValues[3] <= 0) {
        throw new AssertionError("INT16 inference output did not preserve expected signs.");
      }
    }

    return "PASS: CompiledModel INT16 ShortArray I/O and inference succeeded.";
  }

  private static void assertInt16(TensorType type, String tensorName) {
    if (type.getElementType() != TensorType.ElementType.INT16) {
      throw new AssertionError(tensorName + " has type " + type.getElementType() + ", not INT16.");
    }
  }
}
