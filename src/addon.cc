#include "Module.h"
#include "Tensor.h"
#include "Tokenizer.h"
#include <napi.h>

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports = executorch::node::Tensor::Init(env, exports);
  exports = executorch::node::Module::Init(env, exports);
  exports = executorch::node::Tiktoken::Init(env, exports);
  exports = executorch::node::BPETokenizer::Init(env, exports);
  return exports;
}

NODE_API_MODULE(executorch, Init)
