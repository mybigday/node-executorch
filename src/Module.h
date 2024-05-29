#pragma once

#include <executorch/extension/module/module.h>
#include <memory>
#include <napi.h>

namespace executorch {
namespace node {

// Wrap torch::executor::Module to avoid memory leak
class ModuleHolder {
public:
  ModuleHolder(torch::executor::Module *module) { module_.reset(module); }

  torch::executor::Module *operator->() { return module_.get(); }

  torch::executor::Module &operator*() { return *module_; }

private:
  std::unique_ptr<torch::executor::Module> module_;
};

class Module : public Napi::ObjectWrap<Module> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  static inline Napi::Object New(Napi::External<ModuleHolder> module) {
    return constructor.New({module});
  }

  Module(const Napi::CallbackInfo &info);

protected:
  static Napi::Value Load(const Napi::CallbackInfo &info);
  Napi::Value LoadMethod(const Napi::CallbackInfo &info);
  Napi::Value Forward(const Napi::CallbackInfo &info);
  Napi::Value Execute(const Napi::CallbackInfo &info);
  Napi::Value MethodNames(const Napi::CallbackInfo &info);
  void Dispose(const Napi::CallbackInfo &info);

private:
  static Napi::FunctionReference constructor;
  std::unique_ptr<ModuleHolder> module_;
};

} // namespace node
} // namespace executorch
