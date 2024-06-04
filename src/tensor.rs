use neon::types::Finalize;
use cpp::{cpp, cpp_class};

cpp! {{
  #include <executorch/runtime/core/exec_aten/exec_aten.h>
}}

pub enum TensorType {
    UInt8 = 0,
    Int8 = 1,
    Int16 = 2,
    Int32 = 3,
    Int64 = 4,
    // Float16 = 5,
    Float32 = 6,
    Float64 = 7,
    Bool = 11,
}

impl From<i32> for TensorType {
    fn from(value: i32) -> Self {
        match value {
            0 => TensorType::UInt8,
            1 => TensorType::Int8,
            2 => TensorType::Int16,
            3 => TensorType::Int32,
            4 => TensorType::Int64,
            6 => TensorType::Float32,
            7 => TensorType::Float64,
            11 => TensorType::Bool,
            _ => panic!("Invalid dtype"),
        }
    }
}

cpp_class!(unsafe struct AtenTensor as "exec_aten::Tensor");

impl AtenTensor {
    fn new<T>(dtype: TensorType, dim: i64, shape: *mut i32, data: *mut T) -> Self {
        let dtype_num = dtype as i32;
        unsafe {
            cpp!([dtype_num as "int32_t", dim as "ssize_t", shape as "int32_t*", data as "void*"] -> AtenTensor as "exec_aten::Tensor" {
                auto tensor_impl = new exec_aten::TensorImpl(
                    static_cast<exec_aten::ScalarType>(dtype_num),
                    dim,
                    shape,
                    data
                );
                return exec_aten::Tensor(tensor_impl);
            })
        }
    }

    fn dim(&self) -> i64 {
        unsafe {
            cpp!([self as "exec_aten::Tensor*"] -> i64 as "ssize_t" {
                return self->dim();
            })
        }
    }
}

pub struct Tensor<T> {
    tensor: AtenTensor,
    data: Vec<T>,
    shape: Vec<i32>,
}

impl<T> Tensor<T> {
    pub fn dim(&self) -> i64 {
        self.tensor.dim()
    }
}

impl<T> Finalize for Tensor<T> {}

// u8
impl Tensor<u8> {
    pub fn new(mut shape: Vec<i32>, mut  data: Vec<u8>) -> Self {
        let dim = shape.len() as i64;
        let shape_ptr = shape.as_mut_ptr();
        let data_ptr = data.as_mut_ptr();
        let tensor = AtenTensor::new(TensorType::UInt8, dim, shape_ptr, data_ptr);
        Tensor {
            tensor,
            data,
            shape,
        }
    }
}
