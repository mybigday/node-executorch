#include "Tensor.h"
#include "utils.h"
#include <string>
#include <unordered_map>

namespace executorch::node {

const std::unordered_map<exec_aten::ScalarType, size_t> dtypeSize = {
    {exec_aten::ScalarType::Byte, 1},
    {exec_aten::ScalarType::Char, 1},
    {exec_aten::ScalarType::Short, 2},
    {exec_aten::ScalarType::Int, 4},
    {exec_aten::ScalarType::Long, 8},
    {exec_aten::ScalarType::Float, 4},
    {exec_aten::ScalarType::Double, 8},
    {exec_aten::ScalarType::Bool, 1}};

const std::unordered_map<std::string, exec_aten::ScalarType> dtypeMap = {
    {"uint8", exec_aten::ScalarType::Byte},
    {"int8", exec_aten::ScalarType::Char},
    {"int16", exec_aten::ScalarType::Short},
    {"int32", exec_aten::ScalarType::Int},
    {"int64", exec_aten::ScalarType::Long},
    {"float32", exec_aten::ScalarType::Float},
    {"float64", exec_aten::ScalarType::Double},
    {"bool", exec_aten::ScalarType::Bool}};

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

void *getData(const Napi::Value &value, size_t size) {
  if (value.IsBuffer()) {
    Napi::Buffer<uint8_t> buffer = value.As<Napi::Buffer<uint8_t>>();
    if (buffer.Length() != size) {
      throw std::runtime_error("Invalid buffer size");
    }
    char *data = new char[size];
    memcpy(data, buffer.Data(), size);
    return data;
  } else if (value.IsTypedArray()) {
    Napi::TypedArray typedArray = value.As<Napi::TypedArray>();
    if (typedArray.ByteLength() != size) {
      throw std::runtime_error("Invalid typed array size");
    }
    char *data = new char[size];
    memcpy(data, typedArray.ArrayBuffer().Data(), size);
    return data;
  } else {
    throw std::runtime_error("Unsupported data type");
  }
}

size_t calcSize(exec_aten::ScalarType type, size_t rank, exec_aten::SizesType *dims) {
  size_t size = dtypeSize.at(type);
  for (size_t i = 0; i < rank; i++) {
    size *= dims[i];
  }
  return size;
}

Tensor::Tensor(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<Tensor>(info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (info.Length() == 0) {
    return;
  }

  if (info.Length() < 3) {
    Napi::TypeError::New(env, "Expected 3 arguments")
        .ThrowAsJavaScriptException();
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
  exec_aten::SizesType *dims = new exec_aten::SizesType[rank];
  for (size_t i = 0; i < rank; i++) {
    Napi::Value value = jsDims.Get(i);
    if (!value.IsNumber()) {
      Napi::TypeError::New(env, "Expected number").ThrowAsJavaScriptException();
      return;
    }
    dims[i] = value.ToNumber().Int32Value();
  }

  try {
    auto type = getType(dtype);
    tensor_ = std::make_unique<exec_aten::Tensor>(new exec_aten::TensorImpl(
        getType(dtype), rank, dims, getData(info[2], calcSize(type, rank, dims))));
  } catch (std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
  }
}

size_t getSlicePos(Napi::Value val, size_t dimSize, size_t default_value) {
  if (val.IsNumber()) {
    auto num = val.ToNumber().Int32Value();
    if (num < 0)
      num += dimSize;
    if (num < 0 || num >= dimSize) {
      throw std::runtime_error("Index out of range");
    }
    return num;
  } else {
    return default_value;
  }
}

Napi::Value Tensor::Shape(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (tensor_ == nullptr) {
    Napi::TypeError::New(env, "Tensor is disposed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  Napi::Array shape = Napi::Array::New(env, tensor_->dim());
  for (size_t i = 0; i < tensor_->dim(); i++) {
    shape.Set(i, Napi::Number::New(env, tensor_->size(i)));
  }

  return shape;
}

Napi::Value Tensor::Dtype(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  return Napi::String::New(env, getTypeName(tensor_->scalar_type()));
}

Napi::Value Tensor::GetData(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (tensor_ == nullptr) {
    Napi::TypeError::New(env, "Tensor is disposed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  size_t size = tensor_->nbytes();
  size_t n_elem = tensor_->numel();
  const void *data = tensor_->const_data_ptr();

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
    auto boolData = static_cast<const bool *>(data);
    for (size_t i = 0; i < n_elem; i++) {
      array.Set(i, Napi::Boolean::New(env, boolData[i]));
    }
    return array;
  } break;
  default:
    throw std::runtime_error("Unsupported dtype");
  }
}

void Tensor::SetData(const Napi::CallbackInfo &info, const Napi::Value &value) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  auto data = getData(value, tensor_->nbytes());
  memcpy(tensor_->mutable_data_ptr(), data, tensor_->nbytes());
}

void Tensor::SetIndex(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (tensor_ == nullptr) {
    Napi::TypeError::New(env, "Tensor is disposed").ThrowAsJavaScriptException();
    return;
  }

  if (info.Length() < 2) {
    Napi::TypeError::New(env, "Expected 2 arguments")
        .ThrowAsJavaScriptException();
    return;
  }
  if (!info[0].IsArray()) {
    Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
    return;
  }
  if (!info[1].IsNumber() && !info[1].IsBoolean()) {
    Napi::TypeError::New(env, "Expected number, string or boolean")
        .ThrowAsJavaScriptException();
    return;
  }

  size_t pos = 0;

  Napi::Array jsPosition = info[0].As<Napi::Array>();
  size_t rank = tensor_->dim();
  if (jsPosition.Length() != rank) {
    Napi::TypeError::New(env, "Invalid position").ThrowAsJavaScriptException();
    return;
  }

  for (size_t i = 0; i < rank; i++) {
    Napi::Value value = jsPosition.Get(i);
    if (!value.IsNumber()) {
      Napi::TypeError::New(env, "Expected number").ThrowAsJavaScriptException();
      return;
    }
    pos += value.ToNumber().Int32Value() * (i == 0 ? 1 : tensor_->size(i - 1));
  }

  void *data = tensor_->mutable_data_ptr();
  switch (tensor_->scalar_type()) {
  case exec_aten::ScalarType::Byte:
    reinterpret_cast<uint8_t *>(data)[pos] =
        (uint8_t)info[1].ToNumber().Int32Value();
    break;
  case exec_aten::ScalarType::Char:
    reinterpret_cast<int8_t *>(data)[pos] =
        (int8_t)info[1].ToNumber().Int32Value();
    break;
  case exec_aten::ScalarType::Short:
    reinterpret_cast<int16_t *>(data)[pos] =
        (int16_t)info[1].ToNumber().Int32Value();
    break;
  case exec_aten::ScalarType::Int:
    reinterpret_cast<int32_t *>(data)[pos] = info[1].ToNumber().Int32Value();
    break;
  case exec_aten::ScalarType::Long:
    reinterpret_cast<int64_t *>(data)[pos] = info[1].ToNumber().Int64Value();
    break;
  case exec_aten::ScalarType::Float:
    reinterpret_cast<float *>(data)[pos] = info[1].ToNumber().FloatValue();
    break;
  case exec_aten::ScalarType::Double:
    reinterpret_cast<double *>(data)[pos] = info[1].ToNumber().DoubleValue();
    break;
  case exec_aten::ScalarType::Bool:
    reinterpret_cast<bool *>(data)[pos] = info[1].ToBoolean().Value();
    break;
  default:
    throw std::runtime_error("Unsupported dtype");
  }
}

Napi::Value Tensor::Slice(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (tensor_ == nullptr) {
    Napi::TypeError::New(env, "Tensor is disposed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  auto slicePos = info[0].As<Napi::Array>();
  size_t rank = tensor_->dim();
  if (slicePos.Length() != rank) {
    Napi::TypeError::New(env, "Invalid position").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  size_t n_elem = 1;

  std::vector<size_t> startVec(rank);
  std::vector<size_t> endVec(rank);
  for (size_t i = 0; i < rank; i++) {
    Napi::Value sliceDim = slicePos.Get(i);
    auto dimSize = tensor_->size(i);
    if (sliceDim.IsArray()) {
      Napi::Array dim = sliceDim.As<Napi::Array>();
      if (dim.Length() != 2) {
        Napi::TypeError::New(env, "Invalid position")
            .ThrowAsJavaScriptException();
        return env.Undefined();
      }
      startVec[i] = getSlicePos(dim.Get(Napi::Number::New(env, 0)), dimSize, 0);
      endVec[i] =
          getSlicePos(dim.Get(Napi::Number::New(env, 1)), dimSize, dimSize);
    } else {
      startVec[i] = 0;
      endVec[i] = dimSize;
    }
    n_elem *= endVec[i] - startVec[i];
  }

  ssize_t elem_size = tensor_->element_size();
  char *newData = new char[n_elem * elem_size];

  const char *data = (char*) tensor_->const_data_ptr();

  for (size_t i = 0; i < n_elem; i++) {
    size_t offset = 0;
    size_t pos = i;
    for (size_t j = 0; j < rank; j++) {
      size_t stride = tensor_->size(j) - 1;
      size_t dim_size = endVec[j] - startVec[j];
      size_t dim_pos = pos % dim_size;
      pos /= dim_size;
      offset += (startVec[j] + dim_pos) * stride;
    }
    memcpy(newData + i * elem_size,
           data + offset * elem_size,
           elem_size);
  }

  auto dims = new exec_aten::SizesType[rank];
  for (size_t i = 0; i < rank; i++) {
    dims[i] = endVec[i] - startVec[i];
  }
  exec_aten::Tensor tensor(
      new exec_aten::TensorImpl(tensor_->scalar_type(), rank, dims, newData));
  return Tensor::New(tensor);
}

Napi::Value Tensor::Concat(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (info.Length() < 2 || !info[1].IsNumber()) {
    Napi::TypeError::New(env, "Expected number").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  auto js_tensors = info[0].As<Napi::Array>();
  size_t n_tensors = js_tensors.Length();
  if (n_tensors == 0) {
    Napi::TypeError::New(env, "Expected non-empty array")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  size_t axis = info.Length() > 1 ? info[1].ToNumber().Int32Value() : 0;
  std::vector<exec_aten::Tensor *> tensors(n_tensors);
  std::vector<size_t> sizes;
  size_t rank = 0;
  exec_aten::ScalarType dtype;

  for (size_t i = 0; i < n_tensors; i++) {
    auto item = js_tensors.Get(i);
    if (!Tensor::IsInstance(item)) {
      Napi::TypeError::New(env,
                           "Item " + std::to_string(i) + " is not a Tensor")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
    auto tensor = Napi::ObjectWrap<Tensor>::Unwrap(item.As<Napi::Object>())->
        GetTensorPtr();
    if (tensor == nullptr) {
      Napi::TypeError::New(env, "Tensor is disposed").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    tensors[i] = tensor;
    if (i == 0) {
      dtype = tensor->scalar_type();
      rank = tensor->dim();
      sizes.resize(rank);
      for (size_t j = 0; j < rank; j++) {
        sizes[j] = tensor->size(j);
      }
      if (axis >= rank) {
        Napi::TypeError::New(env, "Invalid axis").ThrowAsJavaScriptException();
        return env.Undefined();
      }
    } else if (dtype != tensor->scalar_type()) {
      Napi::TypeError::New(env, "Tensors have different dtypes")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    } else if (rank != tensor->dim()) {
      Napi::TypeError::New(env, "Tensors have different ranks")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    } else {
      for (size_t j = 0; j < rank; j++) {
        if (j == axis) {
          sizes[j] += tensor->size(j);
          continue;
        }
        if (sizes[j] != tensor->size(j) && j != axis) {
          Napi::TypeError::New(env, "Tensors have different sizes")
              .ThrowAsJavaScriptException();
          return env.Undefined();
        }
      }
    }
  }

  size_t n_elem = 1;
  for (size_t i = 0; i < rank; i++) {
    n_elem *= sizes[i];
  }
  ssize_t elem_size = tensors[0]->element_size();
  char *newData = new char[n_elem * elem_size];

  size_t trip_step = 1;
  for (size_t j = 0; j < axis; j++) {
    trip_step *= tensors[0]->size(j);
  }

  size_t chunk_size = elem_size;
  for (size_t k = axis; k < rank; k++) {
    chunk_size *= tensors[0]->size(k);
  }

  for (size_t i = 0; i < trip_step; i++) {
    for (size_t j = 0; j < n_tensors; j++) {
      const char *data = (char*) tensors[j]->const_data_ptr();
      memcpy(newData + j * chunk_size +
                 i * n_tensors * chunk_size,
             data + chunk_size * i,
             chunk_size);
    }
  }

  auto *dims = new exec_aten::SizesType[rank];
  for (size_t i = 0; i < rank; i++) {
    dims[i] = sizes[i];
  }
  exec_aten::Tensor tensor(
      new exec_aten::TensorImpl(dtype, rank, dims, newData));
  return Tensor::New(tensor);
}

Napi::Value Tensor::Reshape(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  if (tensor_ == nullptr) {
    Napi::TypeError::New(env, "Tensor is disposed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "Expected array").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  auto jsDims = info[0].As<Napi::Array>();
  size_t rank = jsDims.Length();
  exec_aten::SizesType *dims = new exec_aten::SizesType[rank];
  size_t n_elem = 1;
  for (size_t i = 0; i < rank; i++) {
    Napi::Value value = jsDims.Get(i);
    if (!value.IsNumber()) {
      Napi::TypeError::New(env, "Expected number").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    dims[i] = value.ToNumber().Int32Value();
    n_elem *= dims[i];
  }

  if (n_elem != tensor_->numel()) {
    Napi::TypeError::New(env, "Invalid shape").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  tensor_ = std::make_unique<exec_aten::Tensor>(
      new exec_aten::TensorImpl(tensor_->scalar_type(), rank, dims, tensor_->mutable_data_ptr()));
  return info.This();
}

void Tensor::Dispose(const Napi::CallbackInfo &info) {
  tensor_.reset();
}

Napi::Object Tensor::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function func =
      DefineClass(env, "Tensor",
                  {StaticMethod("concat", &Tensor::Concat),
                   InstanceAccessor("shape", &Tensor::Shape, nullptr),
                   InstanceAccessor("dtype", &Tensor::Dtype, nullptr),
                   InstanceAccessor("data", &Tensor::GetData, &Tensor::SetData),
                   InstanceMethod("setIndex", &Tensor::SetIndex),
                   InstanceMethod("slice", &Tensor::Slice),
                   InstanceMethod("reshape", &Tensor::Reshape),
                    InstanceMethod("dispose", &Tensor::Dispose)});

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set("Tensor", func);

  return exports;
}

Napi::FunctionReference Tensor::constructor;

} // namespace executorch::node
