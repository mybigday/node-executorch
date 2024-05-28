#include <string>
#include <napi.h>
#include <executorch/runtime/core/evalue.h>

torch::executor::EValue evalueFromNapiValue(const Napi::Value &value);

Napi::Value napiValueFromEValue(const Napi::Env &env,
                                 const torch::executor::EValue &evalue);

std::string errorString(const torch::executor::Error &error);
