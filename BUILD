cc_library(
    name = "utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.h"],
)

cc_library(
  name = "ggml",
  srcs = ["ggml.c"],
  hdrs = ["ggml.h"],
)

cc_binary(
  name = "chat",
  srcs = ["chat.cpp"],
  deps = [":ggml", ":utils"],
  copts = ["-O3 -DNDEBUG -mavx -mavx2 -mf16c -mfma"],
)
