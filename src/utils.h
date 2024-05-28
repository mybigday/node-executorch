#include <executorch/runtime/core/evalue.h>
#include <napi.h>
#include <string>

torch::executor::EValue evalueFromNapiValue(const Napi::Value &value);

Napi::Value napiValueFromEValue(const Napi::Env &env,
                                const torch::executor::EValue &evalue);

std::string errorString(const torch::executor::Error &error);
