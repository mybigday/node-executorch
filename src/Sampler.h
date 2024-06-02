#pragma once

#include <executorch/examples/models/llama2/sampler/sampler.h>
#include <memory>
#include <napi.h>

namespace executorch {
namespace node {

class Sampler : public Napi::ObjectWrap<Sampler> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  Sampler(const Napi::CallbackInfo &info);

protected:
  Napi::Value Sample(const Napi::CallbackInfo &info);
  void Dispose(const Napi::CallbackInfo &info);

  static Napi::Value Concat(const Napi::CallbackInfo &info);

private:
  static Napi::FunctionReference constructor;
  int32_t vocab_size_ = 0;
  std::unique_ptr<torch::executor::Sampler> sampler_ = nullptr;
};

} // namespace node
} // namespace executorch
