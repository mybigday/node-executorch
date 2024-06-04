pub mod sampler;
pub mod tensor;
pub mod macros;

use neon::prelude::*;
use sampler::*;
use tensor::*;

fn create_u8_tensor(mut cx: FunctionContext) -> JsResult<JsBox<Tensor::<u8>>> {
    let shape = get_arg_as_vec!(cx, 0, JsNumber, i32);
    let data = get_arg_as_vec!(cx, 1, JsNumber, u8);
    Ok(cx.boxed(Tensor::<u8>::new(shape, data)))
}

fn create_sampler(mut cx: FunctionContext) -> JsResult<JsBox<Sampler>> {
    let vocab_size = cx.argument::<JsNumber>(0)?.value(&mut cx) as i32;
    let temperature = cx.argument::<JsNumber>(1)?.value(&mut cx) as f32;
    let topp = cx.argument::<JsNumber>(2)?.value(&mut cx) as f32;
    let rng_seed = cx.argument::<JsNumber>(3)?.value(&mut cx) as u64;
    Ok(cx.boxed(Sampler::new(vocab_size, temperature, topp, rng_seed)))
}

fn sampler_sample(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let sampler = cx.argument::<JsBox<Sampler>>(0)?;
    let inputs: Vec<f32> = cx.argument::<JsArray>(1)?
        .to_vec(&mut cx)?
        .iter()
        .map(|val| val.downcast::<JsNumber, _>(&mut cx).unwrap().value(&mut cx) as f32)
        .collect();
    Ok(cx.number(sampler.sample(inputs)))
}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("createSampler", create_sampler)?;
    cx.export_function("samplerSample", sampler_sample)?;
    cx.export_function("createU8Tensor", create_u8_tensor)?;
    Ok(())
}
