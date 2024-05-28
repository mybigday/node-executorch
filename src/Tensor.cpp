#include "Tensor.h"
#include "utils.h"
#include <string>
#include <unordered_map>

namespace executorch::node {

const std::unordered_map<std::string, exec_aten::ScalarType> dtypeMap = {
  {"uint8", exec_aten::ScalarType::Byte},
  {"int8", exec_aten::ScalarType::Char},
  {"int16", exec_aten::ScalarType::Short},
  {"int32", exec_aten::ScalarType::Int},
  {"int64", exec_aten::ScalarType::Long},
  {"float32", exec_aten::ScalarType::Float},
  {"float64", exec_aten::ScalarType::Double},
  {"bool", exec_aten::ScalarType::Bool}
};

exec_aten::ScalarType getType(std::string dtype) {
  auto it = dtypeMap.find(dtype);
  if (it == dtypeMap.end()) {
    throw std::runtime_error("Unsupported dtype");
  }
  return it->second;
}

std::string getTypeName(exec_aten::ScalarType type) {
  for (auto &pair : dtypeMap) {
    if (pair.second == type) {
      return pair.first;
    }
  }
  throw std::runtime_error("Unsupported dtype");
}

void* getData(const Napi::Value &value) {
  if (value.IsBuffer()) {
    Napi::Buffer<uint8_t> buffer = value.As<Napi::Buffer<uint8_t>>();
    return buffer.Data();
  } else if (value.IsTypedArray()) {
    Napi::TypedArray typedArray = value.As<Napi::TypedArray>();
    return typedArray.ArrayBuffer().Data();
  } else {
    throw std::runtime_error("Unsupported data type");
  }
}

Tensor::Tensor(const Napi::CallbackInfo& info) : Napi::ObjectWrap<Tensor>(info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (info.Length() == 1 && info[0].IsExternal()) {
    auto *tensor = info[0].As<Napi::External<exec_aten::Tensor>>().Data();
    tensor_ = std::shared_ptr<exec_aten::Tensor>(tensor);
    return;
  }

  if (info.Length() < 3) {
    Napi::TypeError::New(env, "Expected 3 arguments").ThrowAsJavaScriptException();
    return;
  }
  if (!info[0].IsString()) {
    Napi::TypeError::New(env, "Expected string").ThrowAsJavaScriptException();
    return;
  }
  if (!info[1].IsArray()) {
    Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
    return;
  }

  std::string dtype = info[0].As<Napi::String>().Utf8Value();

  Napi::Array jsDims = info[1].As<Napi::Array>();
  size_t rank = jsDims.Length();
  exec_aten::SizesType* dims = new exec_aten::SizesType[rank];
  for (size_t i = 0; i < rank; i++) {
    Napi::Value value = jsDims.Get(i);
    if (!value.IsNumber()) {
      Napi::TypeError::New(env, "Expected number").ThrowAsJavaScriptException();
      return;
    }
    dims[i] = value.ToNumber().Int32Value();
  }

  try {
    tensor_ = std::make_shared<exec_aten::Tensor>(
      new exec_aten::TensorImpl(getType(dtype), rank, dims, getData(info[2]))
    );
  } catch (std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
  }
}

Napi::Value Tensor::Shape(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  Napi::Array shape = Napi::Array::New(env, tensor_->dim());
  for (size_t i = 0; i < tensor_->dim(); i++) {
    shape.Set(i, Napi::Number::New(env, tensor_->size(i)));
  }

  return shape;
}

Napi::Value Tensor::Dtype(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  return Napi::String::New(env, getTypeName(tensor_->scalar_type()));
}

Napi::Value Tensor::GetData(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  size_t size = tensor_->nbytes();
  size_t n_elem = tensor_->numel();
  auto data = tensor_->const_data_ptr();
  
  switch (tensor_->scalar_type()) {
    case exec_aten::ScalarType::Byte: {
      Napi::Uint8Array array = Napi::Uint8Array::New(env, n_elem);
      memcpy(array.Data(), data, size);
      return array;
    } break;
    case exec_aten::ScalarType::Char: {
      Napi::Int8Array array = Napi::Int8Array::New(env, n_elem);
      memcpy(array.Data(), data, size);
      return array;
    } break;
    case exec_aten::ScalarType::Short: {
      Napi::Int16Array array = Napi::Int16Array::New(env, n_elem);
      memcpy(array.Data(), data, size);
      return array;
    } break;
    case exec_aten::ScalarType::Int: {
      Napi::Int32Array array = Napi::Int32Array::New(env, n_elem);
      memcpy(array.Data(), data, size);
      return array;
    } break;
    case exec_aten::ScalarType::Long: {
      Napi::BigInt64Array array = Napi::BigInt64Array::New(env, n_elem);
      memcpy(array.Data(), data, size);
      return array;
    }
    case exec_aten::ScalarType::Float: {
      Napi::Float32Array array = Napi::Float32Array::New(env, n_elem);
      memcpy(array.Data(), data, size);
      return array;
    } break;
    case exec_aten::ScalarType::Double: {
      Napi::Float64Array array = Napi::Float64Array::New(env, n_elem);
      memcpy(array.Data(), data, size);
      return array;
    } break;
    case exec_aten::ScalarType::Bool: {
      Napi::Array array = Napi::Array::New(env, n_elem);
      auto boolData = static_cast<const bool*>(data);
      for (size_t i = 0; i < n_elem; i++) {
        array.Set(i, Napi::Boolean::New(env, boolData[i]));
      }
      return array;
    } break;
    default:
      throw std::runtime_error("Unsupported dtype");
  }
}

void Tensor::SetData(const Napi::CallbackInfo& info, const Napi::Value &value) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  auto data = getData(value);
  memcpy(tensor_->mutable_data_ptr(), data, tensor_->nbytes());
}

Napi::Object Tensor::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func = DefineClass(env, "Tensor", {
    InstanceAccessor("shape", &Tensor::Shape, nullptr),
    InstanceAccessor("dtype", &Tensor::Dtype, nullptr),
    InstanceAccessor("data", &Tensor::GetData, &Tensor::SetData)
  });

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("Tensor", func);

  return exports;
}

Napi::FunctionReference Tensor::constructor;

}
