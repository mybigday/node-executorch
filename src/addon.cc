#include "Module.h"
#include "Tensor.h"
#include "Sampler.h"
#include <napi.h>

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports = executorch::node::Tensor::Init(env, exports);
  exports = executorch::node::Module::Init(env, exports);
  exports = executorch::node::Sampler::Init(env, exports);
  return exports;
}

NODE_API_MODULE(executorch, Init)
