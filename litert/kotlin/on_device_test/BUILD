load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(
    default_visibility = ["//visibility:public"],
)

copy_file(
    name = "dynamic_model_asset",
    src = "//litert/test:testdata/dynamic_add_model.tflite",
    out = "assets/dynamic_add.tflite",
)

copy_file(
    name = "static_model_asset",
    src = "//litert/test:testdata/simple_model.tflite",
    out = "assets/simple_add.tflite",
)

android_binary(
    name = "compiled_model_resize_smoke_test_app",
    srcs = ["src/main/java/com/google/ai/edge/litert/resizesmoke/CompiledModelResizeSmokeTestActivity.java"],
    assets = [
        ":dynamic_model_asset",
        ":static_model_asset",
    ],
    assets_dir = "assets",
    custom_package = "com.google.ai.edge.litert.resizesmoke",
    manifest = "src/main/AndroidManifest.xml",
    nocompress_extensions = [".tflite"],
    tags = ["manual"],
    deps = ["//litert/kotlin:litert_compiled_model_api"],
)
