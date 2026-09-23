load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(
    default_visibility = ["//visibility:public"],
)

# Package the existing INT16 test model as an Android asset without adding a
# duplicate model file to the source tree.
copy_file(
    name = "int16_model_asset",
    src = "//litert/test:testdata/single_add_default_a16w8_recipe_quantized.tflite",
    out = "assets/int16_add.tflite",
)

android_binary(
    name = "int16_smoke_test_app",
    srcs = ["src/main/java/com/google/ai/edge/litert/int16smoke/Int16SmokeTestActivity.java"],
    assets = [":int16_model_asset"],
    assets_dir = "assets",
    custom_package = "com.google.ai.edge.litert.int16smoke",
    manifest = "src/main/AndroidManifest.xml",
    nocompress_extensions = [".tflite"],
    tags = ["manual"],
    deps = ["//litert/kotlin:litert_compiled_model_api"],
)
