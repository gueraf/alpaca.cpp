# bazel build :chat && ./bazel-bin/chat

cc_library(
  name = "ggml",
  srcs = ["ggml.c"],
  hdrs = ["ggml.h"],
  visibility = ["//visibility:public"],
  copts = ["-O3 -DNDEBUG -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3"],
)

cc_library(
    name = "structs",
    hdrs = ["structs.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.h"],
    deps = [":structs", ":ggml"],
    visibility = ["//visibility:public"],
)

cc_binary(
  name = "chat",
  srcs = ["chat.cpp"],
  deps = [":ggml", ":utils", ":structs"],
)
