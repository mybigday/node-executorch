#include "utils.h"
#include "Tensor.h"

using namespace torch::executor;

EValue evalueFromNapiValue(const Napi::Value &value) {
  EValue evalue;
  if (value.IsNull() || value.IsUndefined()) {
      evalue.tag = Tag::None;
  } else if (value.IsNumber()) {
    evalue.payload.copyable_union.as_double = value.ToNumber().DoubleValue();
    evalue.tag = Tag::Double;
  } else if (value.IsBoolean()) {
    evalue.payload.copyable_union.as_bool = value.ToBoolean().Value();
    evalue.tag = Tag::Bool;
  } else if (value.IsString()) {
    auto str = value.ToString().Utf8Value();
    char *buf = strdup(str.c_str());
    evalue.payload.copyable_union.as_string = ArrayRef<char>(buf, str.size());
    evalue.tag = Tag::String;
  } else if (executorch::node::Tensor::IsInstance(value)) {
    evalue.payload.as_tensor = Napi::ObjectWrap<executorch::node::Tensor>::Unwrap(value.As<Napi::Object>())->GetTensor();
    evalue.tag = Tag::Tensor;
    return evalue;
  } else {
    throw std::runtime_error("Unsupported value type");
  }
  return evalue;
}

Napi::Value napiValueFromEValue(const Napi::Env &env, const EValue &evalue) {
  switch (evalue.tag) {
    case Tag::None:
      return env.Null();
      break;
    case Tag::Int:
      return Napi::Number::New(env, evalue.payload.copyable_union.as_int);
      break;
    case Tag::Double:
      return Napi::Number::New(env, evalue.payload.copyable_union.as_double);
      break;
    case Tag::Bool:
      return Napi::Boolean::New(env, evalue.payload.copyable_union.as_bool);
      break;
    case Tag::String: {
      auto chars = evalue.payload.copyable_union.as_string;
      std::string str(chars.data(), chars.size());
      return Napi::String::New(env, str);
    } break;
    case Tag::Tensor: {
      auto *tensor = new exec_aten::Tensor(evalue.payload.as_tensor);
      return executorch::node::Tensor::New(Napi::External<exec_aten::Tensor>::New(env, tensor));
    } break;
    case Tag::ListBool: {
      auto list = evalue.payload.copyable_union.as_bool_list;
      auto array = Napi::Array::New(env, list.size());
      for (size_t i = 0; i < list.size(); i++) {
        array.Set(i, list[i]);
      }
      return array;
    } break;
    case Tag::ListDouble: {
      auto list = evalue.payload.copyable_union.as_double_list;
      auto array = Napi::Array::New(env, list.size());
      for (size_t i = 0; i < list.size(); i++) {
        array.Set(i, list[i]);
      }
      return array;
    } break;
    case Tag::ListInt: {
      auto list = evalue.payload.copyable_union.as_int_list.get();
      auto array = Napi::Array::New(env, list.size());
      for (size_t i = 0; i < list.size(); i++) {
        array.Set(i, list[i]);
      }
      return array;
    } break;
    default:
      throw std::runtime_error("Unsupported value type");
  }
}

std::string errorString(const Error &error) {
  switch (error) {
    case Error::Internal:
      return "Internal";
    case Error::InvalidState:
      return "InvalidState";
    case Error::EndOfMethod:
      return "EndOfMethod";
    case Error::NotSupported:
      return "NotSupported";
    case Error::NotImplemented:
      return "NotImplemented";
    case Error::InvalidArgument:
      return "InvalidArgument";
    case Error::InvalidType:
      return "InvalidType";
    case Error::OperatorMissing:
      return "OperatorMissing";
    case Error::NotFound:
      return "NotFound";
    case Error::MemoryAllocationFailed:
      return "MemoryAllocationFailed";
    case Error::AccessFailed:
      return "AccessFailed";
    case Error::InvalidProgram:
      return "InvalidProgram";
    case Error::DelegateInvalidCompatibility:
      return "DelegateInvalidCompatibility";
    case Error::DelegateMemoryAllocationFailed:
      return "DelegateMemoryAllocationFailed";
    case Error::DelegateInvalidHandle:
      return "DelegateInvalidHandle";
    default:
      return "UnknownError";
  }
}
