use crate::create_num_array;
use crate::evalue_tag::EValueTag;
use crate::tensor_type::TensorType;
use cpp::{cpp, cpp_class};
use neon::prelude::*;

cpp! {{
    #include "executorch/runtime/executor/method_meta.h"
}}

pub struct TensorInfo<'a> {
    dtype: TensorType,
    shape: &'a [i32],
}

cpp_class!(pub unsafe struct MethodMeta as "torch::executor::MethodMeta");

impl MethodMeta {
    pub fn name(&self) -> String {
        let c_str = unsafe {
            cpp!([self as "const torch::executor::MethodMeta*"] -> *const libc::c_char as "const char*" {
                return self->name();
            })
        };
        unsafe {
            std::ffi::CStr::from_ptr(c_str)
                .to_str()
                .unwrap()
                .to_string()
        }
    }

    pub fn num_inputs(&self) -> usize {
        println!("num_inputs");
        unsafe {
            cpp!([self as "const torch::executor::MethodMeta*"] -> usize as "size_t" {
                return self->num_inputs();
            })
        }
    }

    pub fn num_outputs(&self) -> usize {
        unsafe {
            cpp!([self as "const torch::executor::MethodMeta*"] -> usize as "size_t" {
                return self->num_outputs();
            })
        }
    }

    pub fn input_tag(&self, index: usize) -> EValueTag {
        unsafe {
            cpp!([self as "const torch::executor::MethodMeta*", index as "size_t"] -> i32 as "int32_t" {
                return static_cast<int32_t>(self->input_tag(index).get());
            })
        }.into()
    }

    pub fn output_tag(&self, index: usize) -> EValueTag {
        unsafe {
            cpp!([self as "const torch::executor::MethodMeta*", index as "size_t"] -> i32 as "int32_t" {
                return static_cast<int32_t>(self->output_tag(index).get());
            })
        }.into()
    }

    pub fn input_tensor_info(&self, index: usize) -> TensorInfo {
        let mut dtype: i32 = 0;
        let mut dim: usize = 0;
        let dtype_ptr: *mut i32 = &mut dtype;
        let dim_ptr: *mut usize = &mut dim;
        let c_shape = unsafe {
            cpp!([
                self as "const torch::executor::MethodMeta*",
                index as "size_t",
                dtype_ptr as "int32_t*",
                dim_ptr as "size_t*"
            ] -> *const i32 as "const int32_t*" {
                auto tensor_info = self->input_tensor_meta(index).get();
                *dtype_ptr = static_cast<int32_t>(tensor_info.scalar_type());
                auto shape = tensor_info.sizes();
                *dim_ptr = shape.size();
                return shape.data();
            })
        };
        TensorInfo {
            dtype: dtype.into(),
            shape: unsafe { std::slice::from_raw_parts(c_shape, dim) },
        }
    }

    pub fn output_tensor_info(&self, index: usize) -> TensorInfo {
        let mut dtype: i32 = 0;
        let mut dim: usize = 0;
        let dtype_ptr: *mut i32 = &mut dtype;
        let dim_ptr: *mut usize = &mut dim;
        let c_shape = unsafe {
            cpp!([
                self as "const torch::executor::MethodMeta*",
                index as "size_t",
                dtype_ptr as "int32_t*",
                dim_ptr as "size_t*"
            ] -> *const i32 as "const int32_t*" {
                auto tensor_info = self->output_tensor_meta(index).get();
                *dtype_ptr = static_cast<int32_t>(tensor_info.scalar_type());
                auto shape = tensor_info.sizes();
                *dim_ptr = shape.size();
                return shape.data();
            })
        };
        TensorInfo {
            dtype: dtype.into(),
            shape: unsafe { std::slice::from_raw_parts(c_shape, dim) },
        }
    }

    pub fn to_js<'cx, C>(&self, cx: &mut C) -> JsResult<'cx, JsObject>
    where
        C: Context<'cx>,
    {
        let obj = cx.empty_object();
        let name = cx.string(self.name());
        obj.set(cx, "name", name)?;
        println!("name: {}", self.name());
        let inputs = cx.empty_array();
        for i in 0..self.num_inputs() {
            let input = cx.empty_object();
            let tag = self.input_tag(i);
            let tag_num = cx.number(tag.clone() as i32);
            input.set(cx, "tag", tag_num)?;
            if tag == EValueTag::Tensor {
                let tensor_info = self.input_tensor_info(i);
                let tensor = cx.empty_object();
                let dtype_num = cx.number(tensor_info.dtype as i32);
                tensor.set(cx, "dtype", dtype_num)?;
                create_num_array!(cx, cx, shape, tensor_info.shape);
                tensor.set(cx, "shape", shape)?;
                input.set(cx, "tensor_info", tensor)?;
            }
            inputs.set(cx, i as u32, input)?;
        }
        obj.set(cx, "inputs", inputs)?;
        let outputs = cx.empty_array();
        for i in 0..self.num_outputs() {
            let output = cx.empty_object();
            let tag = self.output_tag(i);
            let tag_num = cx.number(tag.clone() as i32);
            output.set(cx, "tag", tag_num)?;
            if tag == EValueTag::Tensor {
                let tensor_info = self.output_tensor_info(i);
                let tensor = cx.empty_object();
                let dtype_num = cx.number(tensor_info.dtype as i32);
                tensor.set(cx, "dtype", dtype_num)?;
                create_num_array!(cx, cx, shape, tensor_info.shape);
                tensor.set(cx, "shape", shape)?;
                output.set(cx, "tensor_info", tensor)?;
            }
            outputs.set(cx, i as u32, output)?;
        }
        obj.set(cx, "outputs", outputs)?;
        Ok(obj)
    }
}
