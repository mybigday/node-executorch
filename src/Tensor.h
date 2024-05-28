#pragma once

#include <memory>
#include <napi.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace executorch {
namespace node {

class Tensor: public Napi::ObjectWrap<Tensor> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  Tensor(const Napi::CallbackInfo& info);

  static inline bool IsInstance(const Napi::Value &value) {
    return value.IsObject() && value.As<Napi::Object>().InstanceOf(constructor.Value());
  }

  static Napi::Object New(Napi::External<exec_aten::Tensor> tensor) {
    return constructor.New({tensor});
  }

  inline exec_aten::Tensor GetTensor() {
    return *tensor_;
  }

protected:
  Napi::Value Shape(const Napi::CallbackInfo& info);
  Napi::Value Dtype(const Napi::CallbackInfo& info);
  Napi::Value GetData(const Napi::CallbackInfo& info);
  void SetData(const Napi::CallbackInfo& info, const Napi::Value &value);

private:
  static Napi::FunctionReference constructor;
  std::shared_ptr<exec_aten::Tensor> tensor_;
};

}  // namespace node
}  // namespace executorch
