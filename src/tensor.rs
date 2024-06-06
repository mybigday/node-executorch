use crate::arg_get_value;
use crate::arg_get_value_vec;
use crate::create_num_array;
use crate::tensor_type::TensorType;
use cpp::{cpp, cpp_class};
use neon::prelude::*;
use neon::types::buffer::TypedArray;
use neon::types::Finalize;

cpp! {{
    #include "src/tensor.hpp"
}}

cpp_class!(pub unsafe struct Tensor as "TensorHolder");

impl Finalize for Tensor {}

cpp_class!(pub unsafe struct AtenTensor as "exec_aten::Tensor");

impl From<AtenTensor> for Tensor {
    fn from(tensor: AtenTensor) -> Self {
        unsafe {
            cpp!([tensor as "exec_aten::Tensor"] -> Tensor as "TensorHolder" {
                return TensorHolder(std::move(tensor));
            })
        }
    }
}

impl Tensor {
    fn new(
        dtype: TensorType,
        dim: i64,
        shape: *const i32,
        data: *const u8,
        data_nelem: usize,
    ) -> Self {
        let dtype_num = dtype as i32;
        unsafe {
            cpp!([
                dtype_num as "int32_t",
                dim as "int64_t",
                shape as "const int32_t*",
                data as "const uint8_t*",
                data_nelem as "size_t"
            ] -> Tensor as "TensorHolder" {
                return TensorHolder(dtype_num, dim, shape, data, data_nelem);
            })
        }
    }

    pub fn create(dtype: TensorType, shape: &[i32], data: &[u8]) -> Result<Self, String> {
        let mut numel = usize::from(dtype);
        if numel == 0 {
            return Err("Unsupported dtype".to_string());
        }
        for dim in shape {
            numel *= *dim as usize;
        }
        if numel != data.len() {
            return Err("Data length does not match shape".to_string());
        }
        let dim = shape.len() as i64;
        let shape_ptr = shape.as_ptr();
        let data_ptr = data.as_ptr();
        let data_nelem = data.len();
        Ok(Tensor::new(dtype, dim, shape_ptr, data_ptr, data_nelem))
    }

    fn dim(&self) -> i64 {
        unsafe {
            cpp!([self as "const TensorHolder*"] -> i64 as "int64_t" {
                return self->get_tensor().dim();
            })
        }
    }

    pub fn dtype(&self) -> TensorType {
        let dtype = unsafe {
            cpp!([self as "const TensorHolder*"] -> i32 as "int32_t" {
                return static_cast<int32_t>(self->get_tensor().scalar_type());
            })
        };
        TensorType::from(dtype)
    }

    pub fn sizes(&self) -> &[i32] {
        let dim = self.dim();
        let shape_ptr = unsafe {
            cpp!([self as "const TensorHolder*"] -> *const i32 as "const int32_t*" {
                return self->get_tensor().sizes().begin();
            })
        };
        unsafe { std::slice::from_raw_parts(shape_ptr, dim as usize) }
    }

    pub fn numel(&self) -> i64 {
        unsafe {
            cpp!([self as "const TensorHolder*"] -> i64 as "int64_t" {
                return self->get_tensor().numel();
            })
        }
    }

    fn element_size(&self) -> i64 {
        unsafe {
            cpp!([self as "const TensorHolder*"] -> i64 as "int64_t" {
                return self->get_tensor().element_size();
            })
        }
    }

    fn data_ptr(&self) -> *const u8 {
        unsafe {
            cpp!([self as "const TensorHolder*"] -> *mut u8 as "const uint8_t*" {
                return self->get_tensor().const_data_ptr<uint8_t>();
            })
        }
    }

    pub fn get_data<T: Sized>(&self) -> &[T] {
        let data_ptr = self.data_ptr();
        let data_len = self.nbytes();
        unsafe {
            std::slice::from_raw_parts(data_ptr as *const T, data_len / std::mem::size_of::<T>())
        }
    }

    pub fn set_data<T: Sized>(&self, data: &[T]) {
        let data_ptr = data.as_ptr() as *const u8;
        let data_nelem = data.len() * std::mem::size_of::<T>();
        unsafe {
            cpp!([self as "TensorHolder*", data_ptr as "const uint8_t*", data_nelem as "size_t"] {
                self->set_data(data_ptr, data_nelem);
            })
        }
    }

    fn nbytes(&self) -> usize {
        unsafe {
            cpp!([self as "const TensorHolder*"] -> usize as "size_t" {
                return self->get_tensor().nbytes();
            })
        }
    }

    fn set_value_impl(&self, position: *const i32, value: *const u8, value_size: usize) {
        unsafe {
            cpp!([self as "const TensorHolder*", position as "const int32_t*", value as "const uint8_t*", value_size as "size_t"] {
                size_t index = 0;
                size_t stride = 1;
                for (int i = self->get_tensor().dim() - 1; i >= 0; i--) {
                    index += position[i] * stride;
                    stride *= self->get_tensor().sizes()[i];
                }
                memcpy(self->get_tensor().mutable_data_ptr<uint8_t>() + index * value_size, value, value_size);
            })
        }
    }

    pub fn set_value<T: Sized>(&self, position: &[i32], value: T) {
        let position_ptr = position.as_ptr();
        let value_ptr = &value as *const T as *const u8;
        let value_size = std::mem::size_of::<T>();
        self.set_value_impl(position_ptr, value_ptr, value_size);
    }

    pub fn reshape(&self, shape: &[i32]) -> Result<Self, String> {
        let numel = self.numel();
        let mut shape_prod = 1;
        for dim in shape {
            shape_prod *= *dim as i64;
        }
        if numel != shape_prod {
            return Err("New shape must have the same number of elements".to_string());
        }
        Tensor::create(self.dtype(), shape, self.get_data())
    }

    pub fn slice(&self, slices: Vec<(Option<i32>, Option<i32>)>) -> Result<Self, String> {
        let shape = self.sizes();
        let dim = shape.len();

        // Compute actual start and end indices
        let mut start_indices = Vec::<usize>::with_capacity(dim);

        let new_shape: Vec<i32> = slices
            .iter()
            .zip(shape.iter())
            .map(|((start, end), &len)| {
                let start_idx = match start {
                    Some(idx) => *idx % len,
                    None => 0,
                };
                let end_idx = match end {
                    Some(idx) => *idx % len,
                    None => len,
                };
                start_indices.push(start_idx as usize);
                (end_idx - start_idx) as i32
            })
            .collect();

        let elem_size = self.element_size() as usize;

        let mut new_data =
            Vec::with_capacity(new_shape.iter().product::<i32>() as usize * elem_size);

        let data = self.get_data::<u8>();

        for i in 0..new_data.capacity() / elem_size {
            let mut offset = 0;
            let mut pos = i;
            for j in 0..dim {
                let stride = if j == dim - 1 {
                    1
                } else {
                    shape[j + 1..].iter().product::<i32>() as usize
                };
                let dim_size = new_shape[j] as usize;
                let dim_pos = pos % dim_size;
                pos /= dim_size;
                offset += (start_indices[j] + dim_pos) * stride;
            }
            new_data.extend_from_slice(&data[offset * elem_size..(offset + 1) * elem_size]);
        }

        Tensor::create(self.dtype(), new_shape.as_slice(), &new_data)
    }

    pub fn concat(tensors: Vec<Handle<'_, JsBox<Tensor>>>, axis: i64) -> Result<Self, String> {
        if tensors.len() == 0 {
            return Err("Expected non-empty array of tensors".to_string());
        }
        let axis = axis as usize;

        let dtype = tensors[0].dtype();
        let rank = tensors[0].dim() as usize;
        let sizes = tensors[0].sizes().to_vec();
        let mut new_sizes = sizes.clone();

        if axis >= rank {
            return Err("Invalid axis".to_string());
        }

        for &tensor in tensors.iter().skip(1) {
            if dtype != tensor.dtype() {
                return Err("Tensors have different dtypes".to_string());
            }
            if rank != tensor.dim() as usize {
                return Err("Tensors have different ranks".to_string());
            }
            for j in 0..rank {
                if j == axis {
                    new_sizes[j] += tensor.sizes()[j];
                } else if sizes[j] != tensor.sizes()[j] {
                    return Err("Tensors have different sizes".to_string());
                }
            }
        }

        let elem_size = tensors[0].element_size() as usize;
        let numel: usize = new_sizes.iter().map(|&s| s as usize).product();
        let mut new_data = vec![0u8; numel * elem_size];

        let trip_step: usize = sizes.iter().take(axis).map(|&s| s as usize).product();
        let chunk_size: usize = sizes
            .iter()
            .skip(axis)
            .map(|&s| s as usize)
            .product::<usize>()
            * elem_size;
        let n_tensors = tensors.len();

        for i in 0..trip_step {
            for (j, tensor) in tensors.iter().enumerate() {
                let data = tensor.get_data::<u8>();
                let dst_data_offset = i * n_tensors * chunk_size + j * chunk_size;
                let src_data_offset = i * chunk_size;
                new_data[dst_data_offset..dst_data_offset + chunk_size]
                    .copy_from_slice(&data[src_data_offset..src_data_offset + chunk_size]);
            }
        }

        Tensor::create(dtype, &new_sizes, &new_data)
    }

    pub fn to_js<'cx, C>(&self, cx: &mut C) -> JsResult<'cx, JsObject>
    where
        C: Context<'cx>,
    {
        let dtype = self.dtype();
        let shape = self.sizes();
        let info = cx.empty_object();
        let dtype_num = cx.number(dtype as i32);
        info.set(cx, "dtype", dtype_num)?;
        create_num_array!(cx, cx, js_shape, shape);
        info.set(cx, "shape", js_shape)?;
        let ptr = cx.boxed(self.clone());
        info.set(cx, "ptr", ptr)?;
        Ok(info)
    }
}

// JS interface

pub fn create(mut cx: FunctionContext) -> JsResult<JsBox<Tensor>> {
    let dtype: TensorType = (cx.argument::<JsNumber>(0)?.value(&mut cx) as i32).into();
    let shape: Vec<i32> = arg_get_value_vec!(cx, 1, JsNumber, i32);
    let data = cx.argument::<JsArrayBuffer>(2)?.as_slice(&cx);
    match Tensor::create(dtype, shape.as_slice(), &data) {
        Ok(tensor) => Ok(cx.boxed(tensor)),
        Err(e) => cx.throw_error(e),
    }
}

pub fn get_dtype(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let tensor = cx.argument::<JsBox<Tensor>>(0)?;
    Ok(cx.number(tensor.dtype() as i32))
}

pub fn get_shape(mut cx: FunctionContext) -> JsResult<JsArray> {
    let tensor = cx.argument::<JsBox<Tensor>>(0)?;
    let shape = tensor.sizes();
    let js_shape = cx.empty_array();
    for (i, &dim) in shape.iter().enumerate() {
        let n = cx.number(dim);
        js_shape.set(&mut cx, i as u32, n)?;
    }
    Ok(js_shape)
}

pub fn get_data(mut cx: FunctionContext) -> JsResult<JsArrayBuffer> {
    let tensor = cx.argument::<JsBox<Tensor>>(0)?;
    let data = tensor.get_data::<u8>();
    let js_data = JsArrayBuffer::from_slice(&mut cx, data)?;
    Ok(js_data)
}

pub fn set_data(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let tensor = cx.argument::<JsBox<Tensor>>(0)?;
    let data = cx.argument::<JsArrayBuffer>(1)?.as_slice(&cx);
    tensor.set_data(data);
    Ok(cx.undefined())
}

pub fn set_value(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let tensor = cx.argument::<JsBox<Tensor>>(0)?;
    let position: Vec<i32> = arg_get_value_vec!(cx, 1, JsNumber, i32);
    let position = position.as_slice();
    match tensor.dtype() {
        TensorType::UInt8 => {
            let value = arg_get_value!(cx, 2, JsNumber, u8);
            tensor.set_value(position, value);
        }
        TensorType::Int8 => {
            let value = arg_get_value!(cx, 2, JsNumber, i8);
            tensor.set_value(position, value);
        }
        TensorType::Int16 => {
            let value = arg_get_value!(cx, 2, JsNumber, i16);
            tensor.set_value(position, value);
        }
        TensorType::Int32 => {
            let value = arg_get_value!(cx, 2, JsNumber, i32);
            tensor.set_value(position, value);
        }
        TensorType::Int64 => {
            let value = arg_get_value!(cx, 2, JsNumber, i64);
            tensor.set_value(position, value);
        }
        TensorType::Float32 => {
            let value = arg_get_value!(cx, 2, JsNumber, f32);
            tensor.set_value(position, value);
        }
        TensorType::Float64 => {
            let value = arg_get_value!(cx, 2, JsNumber, f64);
            tensor.set_value(position, value);
        }
        TensorType::Bool => {
            let value = arg_get_value!(cx, 2, JsBoolean, bool);
            tensor.set_value(position, value);
        }
        _ => return cx.throw_error("Unsupported dtype"),
    }
    Ok(cx.undefined())
}

pub fn concat(mut cx: FunctionContext) -> JsResult<JsObject> {
    let tensors: Vec<_> = cx
        .argument::<JsArray>(0)?
        .to_vec(&mut cx)?
        .iter()
        .map(|tensor| tensor.downcast::<JsBox<Tensor>, _>(&mut cx).unwrap())
        .collect();
    let axis = arg_get_value!(cx, 1, JsNumber, i64);
    match Tensor::concat(tensors, axis) {
        Ok(tensor) => tensor.to_js(&mut cx),
        Err(e) => cx.throw_error(e),
    }
}

pub fn slice(mut cx: FunctionContext) -> JsResult<JsObject> {
    let tensor = cx.argument::<JsBox<Tensor>>(0)?;
    let dim = tensor.dim() as u32;
    let mut slices = Vec::<(Option<i32>, Option<i32>)>::with_capacity(dim as usize);
    let slices_js = cx.argument::<JsArray>(1)?;
    let len = slices_js.len(&mut cx);
    for i in 0..len {
        let index = cx.number(i);
        let slice = slices_js.get::<JsValue, _, _>(&mut cx, index)?;
        if slice.is_a::<JsArray, _>(&mut cx) {
            let slice = slice.downcast::<JsArray, _>(&mut cx).unwrap();
            let first = slice.get::<JsValue, _, _>(&mut cx, 0).unwrap();
            let second = slice.get::<JsValue, _, _>(&mut cx, 1).unwrap();
            let start = if first.is_a::<JsNumber, _>(&mut cx) {
                Some(
                    first
                        .downcast::<JsNumber, _>(&mut cx)
                        .unwrap()
                        .value(&mut cx) as i32,
                )
            } else {
                None
            };
            let end = if second.is_a::<JsNumber, _>(&mut cx) {
                Some(
                    second
                        .downcast::<JsNumber, _>(&mut cx)
                        .unwrap()
                        .value(&mut cx) as i32,
                )
            } else {
                None
            };
            slices.push((start, end));
        } else if slice.is_a::<JsNumber, _>(&mut cx) {
            let start = slice
                .downcast::<JsNumber, _>(&mut cx)
                .unwrap()
                .value(&mut cx) as i32;
            slices.push((Some(start), Some(start + 1)));
        } else {
            slices.push((None, None));
        }
    }
    for _ in len..dim {
        slices.push((None, None));
    }
    match tensor.slice(slices) {
        Ok(tensor) => tensor.to_js(&mut cx),
        Err(e) => cx.throw_error(e),
    }
}

pub fn reshape(mut cx: FunctionContext) -> JsResult<JsObject> {
    let tensor = cx.argument::<JsBox<Tensor>>(0)?;
    let shape_vec: Vec<i32> = arg_get_value_vec!(cx, 1, JsNumber, i32);
    match tensor.reshape(shape_vec.as_slice()) {
        Ok(tensor) => tensor.to_js(&mut cx),
        Err(e) => cx.throw_error(e),
    }
}
