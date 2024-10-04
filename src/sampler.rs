use crate::arg_get_value;
use crate::tensor::Tensor;
use crate::tensor_type::TensorType;
use cpp::{cpp, cpp_class};
use neon::prelude::*;
use neon::types::Finalize;

cpp! {{
  #include <executorch/extension/llm/sampler/sampler.h>
  #include <executorch/extension/llm/sampler/sampler.cpp>
}}

cpp_class!(pub unsafe struct Sampler as "torch::executor::Sampler");

impl Finalize for Sampler {}

impl Sampler {
    pub fn new(vocab_size: i32, temperature: f32, topp: f32, rng_seed: u64) -> Self {
        unsafe {
            cpp!([vocab_size as "int", temperature as "float", topp as "float", rng_seed as "uint64_t"] -> Sampler as "torch::executor::Sampler" {
                return torch::executor::Sampler(vocab_size, temperature, topp, rng_seed);
            })
        }
    }

    pub fn sample(&self, param: &[f32]) -> i32 {
        let array: *const f32 = param.as_ptr();
        unsafe {
            cpp!([self as "torch::executor::Sampler*", array as "float *"] -> i32 as "int32_t" {
                return self->sample<float>(array);
            })
        }
    }
}

// JS interface

pub fn create(mut cx: FunctionContext) -> JsResult<JsBox<Sampler>> {
    let vocab_size = arg_get_value!(cx, 0, JsNumber, i32);
    let temperature = arg_get_value!(cx, 1, JsNumber, f32);
    let topp = arg_get_value!(cx, 2, JsNumber, f32);
    let rng_seed = arg_get_value!(cx, 3, JsNumber, u64);
    Ok(cx.boxed(Sampler::new(vocab_size, temperature, topp, rng_seed)))
}

pub fn sample(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let sampler = cx.argument::<JsBox<Sampler>>(0)?;
    let input = cx.argument::<JsBox<Tensor>>(1)?;
    if input.dtype() != TensorType::Float32 {
        return cx.throw_error("Input tensor must be of type Float32");
    }
    let shape = input.sizes();
    if shape.len() != 3 || shape[0] != 1 {
        return cx.throw_error("Input tensor must have shape [1, ?, N]");
    }
    let inputs = input.get_data::<f32>();
    let slice_start: usize = ((shape[1] - 1) * shape[2]) as usize;
    let slice_end: usize = (shape[1] * shape[2]) as usize;
    Ok(cx.number(sampler.sample(inputs[slice_start..slice_end].as_ref())))
}
