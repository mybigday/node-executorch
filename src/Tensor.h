#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <memory>
#include <napi.h>

namespace executorch {
namespace node {

class Tensor : public Napi::ObjectWrap<Tensor> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  Tensor(const Napi::CallbackInfo &info);

  static inline bool IsInstance(const Napi::Value &value) {
    return value.IsObject() &&
           value.As<Napi::Object>().InstanceOf(constructor.Value());
  }

  static Napi::Object New(const exec_aten::Tensor &tensor) {
    auto instance = constructor.New({});
    auto *obj = Napi::ObjectWrap<Tensor>::Unwrap(instance);
    obj->tensor_ = std::make_unique<exec_aten::Tensor>(std::move(tensor.unsafeGetTensorImpl()));
    return instance;
  }

  inline exec_aten::Tensor GetTensor() { return *tensor_; }
  inline exec_aten::Tensor* GetTensorPtr() { return tensor_.get(); }

protected:
  Napi::Value Shape(const Napi::CallbackInfo &info);
  Napi::Value Dtype(const Napi::CallbackInfo &info);
  Napi::Value GetData(const Napi::CallbackInfo &info);
  void SetData(const Napi::CallbackInfo &info, const Napi::Value &value);
  void SetValue(const Napi::CallbackInfo &info);
  Napi::Value Slice(const Napi::CallbackInfo &info);
  Napi::Value Reshape(const Napi::CallbackInfo &info);
  void Dispose(const Napi::CallbackInfo &info);

  static Napi::Value Concat(const Napi::CallbackInfo &info);

private:
  static Napi::FunctionReference constructor;
  std::unique_ptr<exec_aten::Tensor> tensor_ = nullptr;
};

} // namespace node
} // namespace executorch
