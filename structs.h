#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <thread>
#include <vector>

// The default parameters
struct gpt_params {
  int32_t seed = -1; // RNG seed
  int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
  int32_t n_predict = 128;    // new tokens to predict
  int32_t repeat_last_n = 64; // last n tokens to penalize
  int32_t n_ctx = 2048;       // context size

  // sampling parameters
  int32_t top_k = 40;
  float top_p = 0.95f;
  float temp = 0.10f;
  float repeat_penalty = 1.30f;

  int32_t n_batch = 8; // batch size for prompt processing

  std::string model = "ggml-alpaca-7b-q4.bin"; // model path
  std::string prompt;

  bool use_color = true; // use color to distinguish generations and inputs

  bool interactive = true;       // interactive mode
  bool interactive_start = true; // reverse prompt immediately
  std::string antiprompt =
      ""; // string upon seeing which more user input is prompted
};

struct gpt_vocab {
  using id = int32_t;
  using token = std::string;

  std::map<token, id> token_to_id;
  std::map<id, token> id_to_token;
};

// default hparams (LLaMA 7B)
struct llama_hparams {
  int32_t n_vocab = 32000;
  int32_t n_ctx = 512; // this is provided as user input?
  int32_t n_embd = 4096;
  int32_t n_mult = 256;
  int32_t n_head = 32;
  int32_t n_layer = 32;
  int32_t n_rot = 64;
  int32_t f16 = 1;
};

struct llama_layer {
  // normalization
  struct ggml_tensor *attention_norm;

  // attention
  struct ggml_tensor *wq;
  struct ggml_tensor *wk;
  struct ggml_tensor *wv;
  struct ggml_tensor *wo;

  // normalization
  struct ggml_tensor *ffn_norm;

  // ff
  struct ggml_tensor *w1;
  struct ggml_tensor *w2;
  struct ggml_tensor *w3;
};

struct llama_model {
  llama_hparams hparams;

  struct ggml_tensor *tok_embeddings;

  struct ggml_tensor *norm;
  struct ggml_tensor *output;

  std::vector<llama_layer> layers;

  // key + value memory
  struct ggml_tensor *memory_k;
  struct ggml_tensor *memory_v;

  //
  struct ggml_context *ctx;
  std::map<std::string, struct ggml_tensor *> tensors;
};
