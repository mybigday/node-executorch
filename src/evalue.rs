use crate::create_bigint_array;
use crate::create_bool_array;
use crate::create_num_array;
use crate::eterror::ETError;
use crate::evalue_tag::EValueTag;
use crate::tensor::AtenTensor;
use crate::tensor::Tensor;
use cpp::{cpp, cpp_class};
use neon::prelude::*;
use neon::types::JsBigInt;

cpp! {{
    #include "executorch/runtime/core/evalue.h"
    #include "src/evalue.hpp"
}}

cpp_class!(pub unsafe struct EValue as "torch::executor::EValue");

impl EValue {
    // Constructors

    pub fn from_tensor(value: &Tensor) -> Self {
        unsafe {
            cpp!([value as "const TensorHolder*"] -> EValue as "torch::executor::EValue" {
                return torch::executor::EValue(value->get_tensor());
            })
        }
    }

    pub fn from_string(value: String) -> Self {
        let cstr = std::ffi::CString::new(value).unwrap();
        unsafe {
            let cstr_ptr = cstr.as_ptr();
            cpp!([cstr_ptr as "const char*"] -> EValue as "torch::executor::EValue" {
                return torch::executor::EValue(cstr_ptr, strlen(cstr_ptr));
            })
        }
    }

    pub fn from_double(value: f64) -> Self {
        unsafe {
            cpp!([value as "double"] -> EValue as "torch::executor::EValue" {
                return torch::executor::EValue(value);
            })
        }
    }

    pub fn from_int(value: i64) -> Self {
        unsafe {
            cpp!([value as "int64_t"] -> EValue as "torch::executor::EValue" {
                return torch::executor::EValue(value);
            })
        }
    }

    pub fn from_bool(value: bool) -> Self {
        unsafe {
            cpp!([value as "bool"] -> EValue as "torch::executor::EValue" {
                return torch::executor::EValue(value);
            })
        }
    }

    pub fn null() -> Self {
        unsafe {
            cpp!([] -> EValue as "torch::executor::EValue" {
                return torch::executor::EValue();
            })
        }
    }

    // Methods

    pub fn tag(&self) -> EValueTag {
        unsafe {
            cpp!([self as "const torch::executor::EValue*"] -> i32 as "int32_t" {
                return tag_to_int(self->tag);
            })
        }
        .into()
    }

    pub fn get_tensor(&self) -> Tensor {
        let aten_tensor = unsafe {
            cpp!([self as "const torch::executor::EValue*"] -> AtenTensor as "exec_aten::Tensor" {
                return self->toTensor();
            })
        };
        Tensor::from(aten_tensor)
    }

    pub fn get_string(&self) -> &str {
        let len: usize = 0;
        let c_str = unsafe {
            let len_ptr = &len as *const usize;
            cpp!([self as "const torch::executor::EValue*", len_ptr as "size_t*"] -> *const std::os::raw::c_char as "const char*" {
                auto str_view = self->toString();
                *len_ptr = str_view.size();
                return str_view.data();
            })
        };
        unsafe { std::str::from_utf8(std::slice::from_raw_parts(c_str as *const u8, len)).unwrap() }
    }

    pub fn get_double(&self) -> f64 {
        unsafe {
            cpp!([self as "const torch::executor::EValue*"] -> f64 as "double" {
                return self->toDouble();
            })
        }
    }

    pub fn get_int(&self) -> i64 {
        unsafe {
            cpp!([self as "const torch::executor::EValue*"] -> i64 as "int64_t" {
                return self->toInt();
            })
        }
    }

    pub fn get_bool(&self) -> bool {
        unsafe {
            cpp!([self as "const torch::executor::EValue*"] -> bool as "bool" {
                return self->toBool();
            })
        }
    }

    pub fn get_bool_list(&self) -> &[bool] {
        let len: usize = 0;
        let c_data = unsafe {
            let len_ptr = &len as *const usize;
            cpp!([self as "const torch::executor::EValue*", len_ptr as "size_t*"] -> *const bool as "const bool*" {
                auto list = self->toBoolList();
                *len_ptr = list.size();
                return list.data();
            })
        };
        unsafe { std::slice::from_raw_parts(c_data, len) }
    }

    pub fn get_double_list(&self) -> &[f64] {
        let len: usize = 0;
        let c_data = unsafe {
            let len_ptr = &len as *const usize;
            cpp!([self as "const torch::executor::EValue*", len_ptr as "size_t*"] -> *const f64 as "const double*" {
                auto list = self->toDoubleList();
                *len_ptr = list.size();
                return list.data();
            })
        };
        unsafe { std::slice::from_raw_parts(c_data, len) }
    }

    pub fn get_int_list(&self) -> &[i64] {
        let len: usize = 0;
        let c_data = unsafe {
            let len_ptr = &len as *const usize;
            cpp!([self as "const torch::executor::EValue*", len_ptr as "size_t*"] -> *const i64 as "const int64_t*" {
                auto list = self->toIntList();
                *len_ptr = list.size();
                return list.data();
            })
        };
        unsafe { std::slice::from_raw_parts(c_data, len) }
    }

    // JS conversion

    pub fn to_js<'cx, C>(&self, cx: &mut C) -> JsResult<'cx, JsObject>
    where
        C: Context<'cx>,
    {
        let tag = self.tag();
        let js_evalue = cx.empty_object();
        let js_tag = cx.number(tag.clone() as i32);
        js_evalue.set(cx, "tag", js_tag)?;
        let value: Handle<'_, JsValue> = match tag {
            EValueTag::None => cx.null().as_value(cx),
            EValueTag::Tensor => self.get_tensor().to_js(cx)?.as_value(cx),
            EValueTag::String => cx.string(self.get_string()).as_value(cx),
            EValueTag::Double => cx.number(self.get_double()).as_value(cx),
            EValueTag::Int => JsBigInt::from_i64(cx, self.get_int()).as_value(cx),
            EValueTag::Bool => cx.boolean(self.get_bool()).as_value(cx),
            EValueTag::ListBool => {
                let list = self.get_bool_list();
                create_bool_array!(cx, cx, js_list, list);
                js_list.as_value(cx)
            }
            EValueTag::ListDouble => {
                let list = self.get_double_list();
                create_num_array!(cx, cx, js_list, list);
                js_list.as_value(cx)
            }
            EValueTag::ListInt => {
                let list = self.get_int_list();
                create_bigint_array!(cx, cx, js_list, list);
                js_list.as_value(cx)
            }
            _ => cx.undefined().as_value(cx),
        };
        js_evalue.set(cx, "data", value)?;
        Ok(js_evalue)
    }

    pub fn from_js<'cx, C>(
        cx: &mut FunctionContext,
        js_evalue: Handle<'cx, JsObject>,
    ) -> Result<Self, ETError>
    where
        C: Context<'cx>,
    {
        if let Ok(tag) = js_evalue.get::<JsNumber, _, _>(cx, "tag") {
            match tag.value(cx) as i32 {
                0 => Ok(Self::null()),
                1 => {
                    if let Ok(value) = js_evalue.get::<JsBox<Tensor>, _, _>(cx, "data") {
                        Ok(Self::from_tensor(&value))
                    } else {
                        Err(ETError::InvalidArgument)
                    }
                }
                2 => {
                    if let Ok(value) = js_evalue.get::<JsString, _, _>(cx, "data") {
                        Ok(Self::from_string(value.value(cx)))
                    } else {
                        Err(ETError::InvalidArgument)
                    }
                }
                3 => {
                    if let Ok(value) = js_evalue.get::<JsNumber, _, _>(cx, "data") {
                        Ok(Self::from_double(value.value(cx)))
                    } else {
                        Err(ETError::InvalidArgument)
                    }
                }
                4 => {
                    if let Ok(value) = js_evalue.get::<JsBigInt, _, _>(cx, "data") {
                        if let Ok(value) = value.to_i64(cx) {
                            Ok(Self::from_int(value))
                        } else {
                            Err(ETError::InvalidArgument)
                        }
                    } else {
                        Err(ETError::InvalidArgument)
                    }
                }
                5 => {
                    if let Ok(value) = js_evalue.get::<JsBoolean, _, _>(cx, "data") {
                        Ok(Self::from_bool(value.value(cx)))
                    } else {
                        Err(ETError::InvalidArgument)
                    }
                }
                _ => Err(ETError::NotSupported),
            }
        } else {
            Err(ETError::InvalidArgument)
        }
    }
}
