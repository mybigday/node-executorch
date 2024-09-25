mod eterror;
mod evalue;
mod evalue_tag;
mod macros;
mod method_meta;
mod module;
mod tensor;
mod tensor_type;

use neon::prelude::*;

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    // Tensor
    cx.export_function("createTensor", tensor::create)?;
    cx.export_function("tensorGetDtype", tensor::get_dtype)?;
    cx.export_function("tensorGetShape", tensor::get_shape)?;
    cx.export_function("tensorGetData", tensor::get_data)?;
    cx.export_function("tensorSetData", tensor::set_data)?;
    cx.export_function("tensorSetValue", tensor::set_value)?;
    cx.export_function("tensorConcat", tensor::concat)?;
    cx.export_function("tensorSlice", tensor::slice)?;
    cx.export_function("tensorReshape", tensor::reshape)?;
    // Module
    cx.export_function("moduleLoad", module::load)?;
    cx.export_function("moduleLoadMethod", module::load_method)?;
    cx.export_function("moduleExecute", module::execute)?;
    cx.export_function("moduleGetMethodMeta", module::get_method_meta)?;
    cx.export_function("moduleMethodNames", module::method_names)?;
    Ok(())
}
