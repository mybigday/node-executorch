use crate::arg_get_value;
use crate::create_string_array;
use crate::eterror::ETError;
use crate::evalue::EValue;
use crate::method_meta::MethodMeta;
use cpp::{cpp, cpp_class};
use neon::prelude::*;
use neon::types::Finalize;

cpp! {{
    #include "src/module.hpp"
}}

cpp_class!(pub unsafe struct Module as "ModuleHolder");

impl Finalize for Module {}

impl Module {
    pub fn copy(&self) -> Self {
        unsafe {
            cpp!([self as "ModuleHolder*"] -> Module as "ModuleHolder" {
                return *self;
            })
        }
    }

    pub fn new(path: String) -> Self {
        let cpath = std::ffi::CString::new(path.as_bytes()).unwrap();
        unsafe {
            let cpath_ptr = cpath.as_ptr();
            cpp!([cpath_ptr as "const char*"] -> Module as "ModuleHolder" {
                return ModuleHolder(std::string(cpath_ptr));
            })
        }
    }

    pub fn load(&self) -> Result<(), ETError> {
        let code = unsafe {
            cpp!([self as "ModuleHolder*"] -> i32 as "int32_t" {
                return static_cast<int32_t>(self->get_module().load());
            })
        };
        match code {
            0 => Ok(()),
            _ => Err(ETError::from(code)),
        }
    }

    pub fn load_method(&self, name: String) -> Result<(), ETError> {
        let cname = std::ffi::CString::new(name).unwrap();
        let code = unsafe {
            let cname_ptr = cname.as_ptr();
            cpp!([self as "ModuleHolder*", cname_ptr as "const char*"] -> i32 as "int32_t" {
                return static_cast<int32_t>(self->get_module().load_method(std::string(cname_ptr)));
            })
        };
        match code {
            0 => Ok(()),
            _ => Err(ETError::from(code)),
        }
    }

    pub fn has_method(&self, method_name: String) -> bool {
        let cname = std::ffi::CString::new(method_name).unwrap();
        let cname_ptr = cname.as_ptr();
        let result = unsafe {
            cpp!([self as "ModuleHolder*", cname_ptr as "const char*"] -> bool as "bool" {
                return self->has_method(std::string(cname_ptr));
            })
        };
        result
    }

    pub fn method_meta(&self, method_name: String) -> Result<MethodMeta, ETError> {
        if !self.has_method(method_name.clone()) {
            return Err(ETError::InvalidArgument);
        }
        let cname = std::ffi::CString::new(method_name).unwrap();
        let cname_ptr = cname.as_ptr();
        let meta = unsafe {
            cpp!([self as "ModuleHolder*", cname_ptr as "const char*"] -> MethodMeta as "torch::executor::MethodMeta" {
                return self->get_module().method_meta(std::string(cname_ptr)).get();
            })
        };
        Ok(meta)
    }

    pub fn method_names(&self) -> Vec<String> {
        let nums: usize = unsafe {
            cpp!([self as "ModuleHolder*"] -> usize as "size_t" {
                return self->method_names().size();
            })
        };
        let mut names = Vec::with_capacity(nums);
        for i in 0..nums {
            let name = unsafe {
                cpp!([self as "ModuleHolder*", i as "size_t"] -> *const libc::c_char as "const char*" {
                    return self->method_names()[i].c_str();
                })
            };
            names.push(unsafe { std::ffi::CStr::from_ptr(name).to_str().unwrap().to_string() });
        }
        names
    }

    pub fn execute(
        &self,
        method_name: String,
        inputs: Vec<EValue>,
    ) -> Result<Vec<EValue>, ETError> {
        let cname = std::ffi::CString::new(method_name).unwrap();
        let cname_ptr = cname.as_ptr();
        let inputs_ptr = inputs.as_ptr();
        let ninputs = inputs.len();
        let nouts = unsafe {
            cpp!([self as "ModuleHolder*", cname_ptr as "const char*"] -> usize as "size_t" {
                auto meta = self->get_module().method_meta(std::string(cname_ptr));
                return meta.ok() ? meta.get().num_outputs() : 0;
            })
        };
        let mut outputs = vec![EValue::null(); nouts];
        let outputs_ptr = outputs.as_mut_ptr();
        let code = unsafe {
            cpp!([
                self as "ModuleHolder*",
                cname_ptr as "const char*",
                inputs_ptr as "const torch::executor::EValue*",
                ninputs as "size_t",
                outputs_ptr as "torch::executor::EValue*"
            ] -> i32 as "int32_t" {
                std::vector<torch::executor::EValue> inputs(inputs_ptr, inputs_ptr + ninputs);
                auto result = self->get_module().execute(std::string(cname_ptr), inputs);
                if (result.ok()) {
                    auto outputs = result.get();
                    for (size_t i = 0; i < outputs.size(); i++) {
                        outputs_ptr[i] = outputs[i];
                    }
                }
                return static_cast<uint32_t>(result.error());
            })
        };
        match code {
            0 => Ok(outputs),
            _ => Err(ETError::from(code)),
        }
    }
}

// JS interface

pub fn load(mut cx: FunctionContext) -> JsResult<JsPromise> {
    let path = arg_get_value!(cx, 0, JsString, String);
    let promise = cx.task(move || path).promise(move |mut cx, path| {
        let module = Module::new(path);
        match module.load() {
            Ok(()) => Ok(cx.boxed(module)),
            Err(e) => cx.throw_error(format!("Error: {:?}", e)),
        }
    });
    Ok(promise)
}

pub fn load_method(mut cx: FunctionContext) -> JsResult<JsPromise> {
    let module = cx.argument::<JsBox<Module>>(0)?.copy();
    let name = arg_get_value!(cx, 1, JsString, String);
    let promise = cx
        .task(move || module.load_method(name))
        .promise(move |mut cx, result| match result {
            Ok(()) => Ok(cx.undefined()),
            Err(e) => cx.throw_error(format!("Failed to load method: {:?}", e)),
        });
    Ok(promise)
}

pub fn execute(mut cx: FunctionContext) -> JsResult<JsPromise> {
    let module = cx.argument::<JsBox<Module>>(0)?.copy();
    // = Arc::new(cx.argument::<JsBox<Module>>(0)?);
    let method_name = arg_get_value!(cx, 1, JsString, String);
    let mut inputs = Vec::<EValue>::new();
    let inputs_js = cx.argument::<JsArray>(2)?;
    for i in 0..inputs_js.len(&mut cx) {
        let input_js = inputs_js.get(&mut cx, i)?;
        match EValue::from_js::<FunctionContext>(&mut cx, input_js) {
            Ok(input) => inputs.push(input),
            Err(e) => return cx.throw_error(format!("Failed to parse input: {:?}", e)),
        }
    }
    let promise = cx
        .task(move || module.execute(method_name, inputs))
        .promise(move |mut cx, result| match result {
            Ok(outputs) => {
                let outputs_js = cx.empty_array();
                for (i, output) in outputs.iter().enumerate() {
                    let output_js = output.to_js(&mut cx)?;
                    outputs_js.set(&mut cx, i as u32, output_js)?;
                }
                Ok(outputs_js)
            }
            Err(e) => cx.throw_error(format!("Failed to execute method: {:?}", e)),
        });
    Ok(promise)
}

pub fn get_method_meta(mut cx: FunctionContext) -> JsResult<JsObject> {
    let module = cx.argument::<JsBox<Module>>(0)?;
    let method_name = arg_get_value!(cx, 1, JsString, String);
    match module.method_meta(method_name) {
        Ok(meta) => meta.to_js(&mut cx),
        Err(e) => cx.throw_error(format!("Failed to get method meta: {:?}", e)),
    }
}

pub fn method_names(mut cx: FunctionContext) -> JsResult<JsArray> {
    let module = cx.argument::<JsBox<Module>>(0)?;
    let names = module.method_names();
    create_string_array!(cx, &mut cx, array, names);
    Ok(array)
}
