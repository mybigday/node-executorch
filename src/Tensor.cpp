#include "common.h"
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
    {"float16", exec_aten::ScalarType::Half},
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

void *getData(const Napi::Env &env, const Napi::Value &value, size_t size) {
  if (value.IsBuffer()) {
    Napi::Buffer<uint8_t> buffer = value.As<Napi::Buffer<uint8_t>>();
    THROW_IF_NOT(env, buffer.Length() == size, "Invalid buffer size");
    char *data = new char[size];
    memcpy(data, buffer.Data(), size);
    return data;
  } else if (value.IsTypedArray()) {
    Napi::TypedArray typedArray = value.As<Napi::TypedArray>();
    THROW_IF_NOT(env, typedArray.ByteLength() == size, "Invalid typed array size");
    char *data = new char[size];
    memcpy(data, typedArray.ArrayBuffer().Data(), size);
    return data;
  } else {
    THROW_IF_NOT(env, false, "Expected buffer or typed array");
  }
  return nullptr;
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

  THROW_IF_NOT(env, info.Length() >= 3, "Expected 3 arguments");
  THROW_IF_NOT(env, info[0].IsString(), "Argument 0 must be a string");
  THROW_IF_NOT(env, info[1].IsArray(), "Argument 1 must be an array");

  std::string dtype = info[0].As<Napi::String>().Utf8Value();

  Napi::Array jsDims = info[1].As<Napi::Array>();
  size_t rank = jsDims.Length();
  exec_aten::SizesType *dims = new exec_aten::SizesType[rank];
  for (size_t i = 0; i < rank; i++) {
    Napi::Value value = jsDims.Get(i);
    THROW_IF_NOT(env, value.IsNumber(), "Dimension must be a number");
    dims[i] = value.ToNumber().Int32Value();
  }

  try {
    auto type = getType(dtype);
    tensor_ = std::make_unique<exec_aten::Tensor>(new exec_aten::TensorImpl(
        getType(dtype), rank, dims, getData(env, info[2], calcSize(type, rank, dims))));
  } catch (std::exception &e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
  }
}

size_t getSlicePos(Napi::Env &env, Napi::Value val, size_t dimSize, size_t default_value) {
  if (val.IsNumber()) {
    auto num = val.ToNumber().Int32Value();
    if (num < 0)
      num += dimSize;
    if (num < 0 || num >= dimSize) {
      Napi::TypeError::New(env, "Index out of range").ThrowAsJavaScriptException();
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

  auto data = getData(env, value, tensor_->nbytes());
  memcpy(tensor_->mutable_data_ptr(), data, tensor_->nbytes());
}

void Tensor::SetValue(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  THROW_IF_NOT(env, tensor_ != nullptr, "Tensor is disposed");
  THROW_IF_NOT(env, info.Length() == 2, "Expected 2 arguments");
  THROW_IF_NOT(env, info[0].IsArray(), "Argument 0 must be an array");
  THROW_IF_NOT(env, info[1].IsNumber() || info[1].IsBoolean(), "Argument 1 must be a number or boolean");

  size_t pos = 0;

  Napi::Array jsPosition = info[0].As<Napi::Array>();
  size_t rank = tensor_->dim();
  THROW_IF_NOT(env, jsPosition.Length() == rank, "Invalid position");

  for (size_t i = 0; i < rank; i++) {
    Napi::Value value = jsPosition.Get(i);
    THROW_IF_NOT(env, value.IsNumber(), "Position must be a number");
    pos += value.ToNumber().Int32Value() * (i == 0 ? 1 : tensor_->size(i - 1));
  }

  void *data = tensor_->mutable_data_ptr();
  switch (tensor_->scalar_type()) {
  case exec_aten::ScalarType::Byte:
    static_cast<uint8_t *>(data)[pos] =
        (uint8_t)info[1].ToNumber().Int32Value();
    break;
  case exec_aten::ScalarType::Char:
    static_cast<int8_t *>(data)[pos] =
        (int8_t)info[1].ToNumber().Int32Value();
    break;
  case exec_aten::ScalarType::Short:
    static_cast<int16_t *>(data)[pos] =
        (int16_t)info[1].ToNumber().Int32Value();
    break;
  case exec_aten::ScalarType::Int:
    static_cast<int32_t *>(data)[pos] = info[1].ToNumber().Int32Value();
    break;
  case exec_aten::ScalarType::Long:
    static_cast<int64_t *>(data)[pos] = info[1].ToNumber().Int64Value();
    break;
  case exec_aten::ScalarType::Float:
    static_cast<float *>(data)[pos] = info[1].ToNumber().FloatValue();
    break;
  case exec_aten::ScalarType::Half:
    static_cast<exec_aten::Half *>(data)[pos] = info[1].ToNumber().FloatValue();
    break;
  case exec_aten::ScalarType::Double:
    static_cast<double *>(data)[pos] = info[1].ToNumber().DoubleValue();
    break;
  case exec_aten::ScalarType::Bool:
    static_cast<bool *>(data)[pos] = info[1].ToBoolean().Value();
    break;
  default:
    throw std::runtime_error("Unsupported dtype");
  }
}

// slice(...slice_position: Array<Optional<Array<Optional<number>>|number>>): Tensor
Napi::Value Tensor::Slice(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  RETURN_IF_NOT(env, tensor_ != nullptr, "Tensor is disposed");
  RETURN_IF_NOT(env, info.Length() >= 1, "Expected at least 1 argument");
  RETURN_IF_NOT(env, info.Length() <= tensor_->dim(), "Invalid position, too many dimensions");

  size_t rank = tensor_->dim();
  size_t n_elem = 1;

  std::vector<size_t> startVec(rank);
  std::vector<size_t> endVec(rank);
  for (size_t i = 0; i < rank; i++) {
    Napi::Value sliceDim = info.Length() > i ? info[i] : env.Undefined();
    auto dimSize = tensor_->size(i);
    if (sliceDim.IsArray()) {
      Napi::Array dim = sliceDim.As<Napi::Array>();
      RETURN_IF_NOT(env, dim.Length() == 2, "Invalid slice position, expected 2 elements");
      startVec[i] = getSlicePos(env, dim.Get(Napi::Number::New(env, 0)), dimSize, 0);
      endVec[i] =
          getSlicePos(env, dim.Get(Napi::Number::New(env, 1)), dimSize, dimSize);
    } else if (sliceDim.IsNumber()) {
      size_t pos = getSlicePos(env, sliceDim, dimSize, 0);
      startVec[i] = pos;
      endVec[i] = pos + 1;
    } else if (sliceDim.IsUndefined() || sliceDim.IsNull()) {
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

// static concat(tensors: Array<Tensor>, axis?: number = 0): Tensor
Napi::Value Tensor::Concat(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  RETURN_IF_NOT(env, info.Length() >= 1, "Expected at least 1 argument");
  RETURN_IF_NOT(env, info[0].IsArray(), "Argument 0 must be an array");
  if (info.Length() > 1) {
    RETURN_IF_NOT(env, info[1].IsNumber(), "Argument 1 must be a number");
  }

  auto js_tensors = info[0].As<Napi::Array>();
  size_t n_tensors = js_tensors.Length();
  
  RETURN_IF_NOT(env, n_tensors > 0, "Expected non-empty array");

  size_t axis = info.Length() > 1 ? info[1].ToNumber().Int32Value() : 0;
  std::vector<exec_aten::Tensor *> tensors(n_tensors);
  std::vector<size_t> sizes;
  size_t rank = 0;
  exec_aten::ScalarType dtype;

  for (size_t i = 0; i < n_tensors; i++) {
    auto item = js_tensors.Get(i);
    RETURN_IF_NOT(env, Tensor::IsInstance(item), "Item is not a Tensor");
    auto tensor = Napi::ObjectWrap<Tensor>::Unwrap(item.As<Napi::Object>())->
        GetTensorPtr();
    RETURN_IF_NOT(env, tensor != nullptr, "Tensor is disposed");
    tensors[i] = tensor;
    if (i == 0) {
      dtype = tensor->scalar_type();
      rank = tensor->dim();
      sizes.resize(rank);
      for (size_t j = 0; j < rank; j++) {
        sizes[j] = tensor->size(j);
      }
      RETURN_IF_NOT(env, axis < rank, "Invalid axis");
    } else if (dtype != tensor->scalar_type()) {
      RETURN_IF_NOT(env, false, "Tensors have different dtypes");
    } else if (rank != tensor->dim()) {
      RETURN_IF_NOT(env, false, "Tensors have different ranks");
    } else {
      for (size_t j = 0; j < rank; j++) {
        if (j == axis) {
          sizes[j] += tensor->size(j);
          continue;
        }
        RETURN_IF_NOT(env, sizes[j] == tensor->size(j) || j == axis, "Tensors have different sizes");
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

  RETURN_IF_NOT(env, tensor_ != nullptr, "Tensor is disposed");
  RETURN_IF_NOT(env, info.Length() == 1, "Expected 1 argument");
  RETURN_IF_NOT(env, info[0].IsArray(), "Argument 0 must be an array");

  auto jsDims = info[0].As<Napi::Array>();
  size_t rank = jsDims.Length();
  exec_aten::SizesType *dims = new exec_aten::SizesType[rank];
  size_t n_elem = 1;
  for (size_t i = 0; i < rank; i++) {
    Napi::Value value = jsDims.Get(i);
    RETURN_IF_NOT(env, value.IsNumber(), "Dimension must be a number");
    dims[i] = value.ToNumber().Int32Value();
    n_elem *= dims[i];
  }
  RETURN_IF_NOT(env, n_elem == tensor_->numel(), "Expected the same number of elements");

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
                   InstanceMethod("setValue", &Tensor::SetValue),
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
