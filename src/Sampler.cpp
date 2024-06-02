#include "common.h"
#include "Sampler.h"
#include "Tensor.h"

namespace executorch::node {

// new Sampler(vocab_size: number, temperature?: number = 0.7, top_p?: number = 0.9, seed?: number = 0): Sampler
Sampler::Sampler(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<Sampler>(info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  THROW_IF_NOT(env, info.Length() >= 1, "Expected at least 1 argument");
  THROW_IF_NOT(env, info[0].IsNumber(), "Argument 0 must be a number");
  if (info.Length() >= 2) {
    THROW_IF_NOT(env, info[1].IsNumber(), "Argument 1 must be a number");
  }
  if (info.Length() >= 3) {
    THROW_IF_NOT(env, info[2].IsNumber(), "Argument 2 must be a number");
  }
  if (info.Length() >= 4) {
    THROW_IF_NOT(env, info[3].IsNumber(), "Argument 3 must be a number");
  }

  vocab_size_ = info[0].As<Napi::Number>().Int32Value();
  double temperature = info.Length() >= 2 ? info[1].As<Napi::Number>().DoubleValue() : 0.7;
  double top_p = info.Length() >= 3 ? info[2].As<Napi::Number>().DoubleValue() : 0.9;
  int64_t seed = info.Length() >= 4 ? info[3].As<Napi::Number>().Int64Value() : 0;

  sampler_ = std::make_unique<torch::executor::Sampler>(vocab_size_, temperature, top_p, seed);
}

// sample(tensor: Tensor): number
Napi::Value Sampler::Sample(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  THROW_IF_NOT(env, sampler_ != nullptr, "Sampler is disposed");

  THROW_IF_NOT(env, info.Length() >= 1, "Expected at least 1 argument");
  THROW_IF_NOT(env, Tensor::IsInstance(info[0]), "Argument 0 must be a Tensor");

  auto tensor = Napi::ObjectWrap<Tensor>::Unwrap(info[0].As<Napi::Object>())->GetTensor();

  // check shape: [1, ?, vocab_size]
  THROW_IF_NOT(env, tensor.dim() == 3, "Expected a 3D tensor");
  THROW_IF_NOT(env, tensor.size(0) == 1, "Batch size must be 1");
  THROW_IF_NOT(env, tensor.size(2) == vocab_size_, "Vocab size mismatch");

  void *data = tensor.mutable_data_ptr();

  if (tensor.size(1) > 1) {
    data = (void *)((char *) data + tensor.element_size() * (tensor.size(1) - 1));
  }

  if (tensor.scalar_type() == exec_aten::ScalarType::Float) {
    auto result = sampler_->sample((float *) data);
    return Napi::Number::New(env, result);
  } else if (tensor.scalar_type() == exec_aten::ScalarType::Half) {
    auto result = sampler_->sample((exec_aten::Half *) data);
  } else {
    THROW_IF_NOT(env, false, "Unsupported tensor type");
  }
  return env.Undefined();
}

void Sampler::Dispose(const Napi::CallbackInfo &info) {
  sampler_.reset();
}

Napi::Object Sampler::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func =
      DefineClass(env, "Sampler",
                  {InstanceMethod("sample", &Sampler::Sample),
                   InstanceMethod("dispose", &Sampler::Dispose)});

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("Sampler", func);

  return exports;
}

Napi::FunctionReference Sampler::constructor;

} // namespace executorch::node
