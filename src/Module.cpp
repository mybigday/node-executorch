#include "Module.h"
#include "utils.h"
#include <string>

namespace executorch::node {

/* LoadWorker */

class LoadWorker : public Napi::AsyncWorker, public Napi::Promise::Deferred {
public:
  LoadWorker(Napi::Env env, const std::string &path)
      : Napi::AsyncWorker(env), Napi::Promise::Deferred(env), path_(path) {}

protected:
  void Execute() {
    try {
      auto *module = new torch::executor::Module(
          path_, torch::executor::Module::MlockConfig::NoMlock);
      module_ = std::make_unique<ModuleHolder>(module);
    } catch (const std::exception &e) {
      SetError(e.what());
    }
  }

  void OnOK() {
    Resolve(Module::New(Napi::External<ModuleHolder>::New(
        Napi::AsyncWorker::Env(), module_.release())));
  }

  void OnError(const Napi::Error &e) { Reject(e.Value()); }

private:
  const std::string path_;
  std::unique_ptr<ModuleHolder> module_;
};

/* ExecuteWorker */
class ExecuteWorker : public Napi::AsyncWorker, public Napi::Promise::Deferred {
public:
  ExecuteWorker(Napi::Env env, ModuleHolder *module, std::string method,
                std::vector<torch::executor::EValue> inputs)
      : Napi::AsyncWorker(env), Napi::Promise::Deferred(env), module_(module),
        method_(method), inputs_(std::move(inputs)) {}

protected:
  void Execute() {
    try {
      auto result = (*module_)->execute(method_, inputs_);
      if (result.ok()) {
        outputs_ = std::move(result.get());
      } else {
        throw std::runtime_error("Failed to execute method: " +
                                 errorString(result.error()));
      }
    } catch (const std::exception &e) {
      SetError(e.what());
    }
  }

  void OnOK() {
    try {
      auto results =
          Napi::Array::New(Napi::AsyncWorker::Env(), outputs_.size());
      for (size_t i = 0; i < outputs_.size(); i++) {
        results.Set(i,
                    napiValueFromEValue(Napi::AsyncWorker::Env(), outputs_[i]));
      }
      Resolve(results);
    } catch (const std::exception &e) {
      Reject(Napi::Error::New(Napi::AsyncWorker::Env(), e.what()).Value());
    }
  }

  void OnError(const Napi::Error &e) { Reject(e.Value()); }

private:
  ModuleHolder *module_;
  const std::string method_;
  const std::vector<torch::executor::EValue> inputs_;
  std::vector<torch::executor::EValue> outputs_;
};

/* LoadMethodWorker */

class LoadMethodWorker : public Napi::AsyncWorker,
                         public Napi::Promise::Deferred {
public:
  LoadMethodWorker(Napi::Env env, ModuleHolder *module, std::string method)
      : Napi::AsyncWorker(env), Napi::Promise::Deferred(env), module_(module),
        method_(method) {}

protected:
  void Execute() {
    try {
      auto error = (*module_)->load_method(method_);
      if (error != torch::executor::Error::Ok) {
        throw std::runtime_error("Failed to load method: " +
                                 errorString(error));
      }
    } catch (const std::exception &e) {
      SetError(e.what());
    }
  }

  void OnOK() { Resolve(Napi::AsyncWorker::Env().Undefined()); }

  void OnError(const Napi::Error &e) { Reject(e.Value()); }

private:
  ModuleHolder *module_;
  const std::string method_;
};

/* Module */

Module::Module(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<Module>(info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (info.Length() < 1 || !info[0].IsExternal()) {
    Napi::TypeError::New(env, "Expected an external")
        .ThrowAsJavaScriptException();
    return;
  }

  auto module = info[0].As<Napi::External<ModuleHolder>>();
  module_.reset(std::move(module.Data()));
}

Napi::Value Module::Execute(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (info.Length() < 2) {
    Napi::TypeError::New(env, "Expected method name and input array")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[1].IsString()) {
    Napi::TypeError::New(env, "Expected method name")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(env, "Expected input array")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  std::string method = info[0].As<Napi::String>().Utf8Value();

  std::vector<torch::executor::EValue> inputs;
  auto inputsArray = info[1].As<Napi::Array>();
  for (size_t i = 0; i < inputsArray.Length(); i++) {
    inputs.push_back(evalueFromNapiValue(inputsArray.Get(i)));
  }

  auto worker = new ExecuteWorker(env, module_.get(), method, inputs);
  worker->Queue();
  return worker->Promise();
}

Napi::Value Module::Load(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "Expected a string").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  std::string path = info[0].As<Napi::String>().Utf8Value();
  auto worker = new LoadWorker(env, path);
  worker->Queue();
  return worker->Promise();
}

Napi::Value Module::LoadMethod(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "Expected a string").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  std::string method = info[0].As<Napi::String>().Utf8Value();

  if ((*module_)->is_method_loaded(method)) {
    return env.Undefined();
  }

  auto worker = new LoadMethodWorker(env, module_.get(), method);
  worker->Queue();
  return worker->Promise();
}

Napi::Value Module::Forward(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "Expected input array")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  std::vector<torch::executor::EValue> inputs;
  auto inputsArray = info[0].As<Napi::Array>();
  for (size_t i = 0; i < inputsArray.Length(); i++) {
    inputs.push_back(evalueFromNapiValue(inputsArray.Get(i)));
  }

  auto worker = new ExecuteWorker(env, module_.get(), "forward", inputs);
  worker->Queue();
  return worker->Promise();
}

Napi::Value Module::MethodNames(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  auto result = (*module_)->method_names();
  if (result.ok()) {
    auto names = result.get();
    auto js_results = Napi::Array::New(env, names.size());
    size_t i = 0;
    for (const auto &name : names) {
      js_results.Set(i++, Napi::String::New(env, name));
    }
    return js_results;
  } else {
    Napi::Error::New(env, "Failed to get method names: " +
                              errorString(result.error()))
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
}

Napi::Object Module::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(
      env, "Module",
      {StaticMethod("load", &Module::Load),
       InstanceAccessor("method_names", &Module::MethodNames, nullptr),
       InstanceMethod("loadMethod", &Module::LoadMethod),
       InstanceMethod("forward", &Module::Forward),
       InstanceMethod("execute", &Module::Execute)});

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("Module", func);

  return exports;
}

Napi::FunctionReference Module::constructor;

} // namespace executorch::node
