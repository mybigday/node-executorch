#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <memory>
#include <vector>

class TensorHolder {
public:
  TensorHolder(size_t dtype, int64_t dim, const exec_aten::SizesType *shape,
               const uint8_t *data, size_t data_nelem) {
    shape_ = std::make_shared<std::vector<exec_aten::SizesType>>(dim);
    memcpy(shape_->data(), shape, dim * sizeof(exec_aten::SizesType));
    data_.reset(malloc(data_nelem), free);
    memcpy(data_.get(), data, data_nelem);
    tensor_ = std::make_shared<exec_aten::Tensor>(
        new exec_aten::TensorImpl(static_cast<exec_aten::ScalarType>(dtype),
                                  dim, shape_->data(), data_.get()));
  }

  TensorHolder(exec_aten::Tensor tensor) {
    tensor_ = std::make_shared<exec_aten::Tensor>(tensor);
  }

  void set_data(const uint8_t *data, size_t data_nelem) {
    data_.reset(malloc(data_nelem), free);
    memcpy(data_.get(), data, data_nelem);
    tensor_->unsafeGetTensorImpl()->set_data(data_.get());
  }

  exec_aten::Tensor &get_tensor() const { return *tensor_; }

private:
  std::shared_ptr<void> data_ = nullptr;
  std::shared_ptr<std::vector<exec_aten::SizesType>> shape_ = nullptr;
  std::shared_ptr<exec_aten::Tensor> tensor_ = nullptr;
};
