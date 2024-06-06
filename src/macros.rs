#[macro_export]
macro_rules! arg_get_value_vec {
    ($cx:ident, $index:expr, $js_elem_type:ty, $elem_type:ty) => {
        $cx.argument::<JsArray>($index)?
            .to_vec(&mut $cx)?
            .iter()
            .map(|value| {
                value
                    .downcast::<$js_elem_type, _>(&mut $cx)
                    .unwrap()
                    .value(&mut $cx) as $elem_type
            })
            .collect()
    };
}

#[macro_export]
macro_rules! arg_get_value {
    ($cx:ident, $index:expr, $js_type:ty, $type:ty) => {
        $cx.argument::<$js_type>($index)?.value(&mut $cx) as $type
    };
}

#[macro_export]
macro_rules! array_get_value {
    ($cx:ident, $array:ident, $index:expr, $js_elem_type:ty, $elem_type:ty) => {
        $array
            .get::<$js_elem_type, _, _>(&mut $cx, $index)?
            .value(&mut $cx) as $elem_type
    };
}

#[macro_export]
macro_rules! create_bool_array {
    ($cx:ident, $mut_cx:expr, $array:ident, $vec:expr) => {
        let $array = $cx.empty_array();
        for (i, &elem) in $vec.iter().enumerate() {
            let elem = $cx.boolean(elem);
            $array.set($mut_cx, i as u32, elem)?;
        }
    };
}

#[macro_export]
macro_rules! create_num_array {
    ($cx:ident, $mut_cx:expr, $array:ident, $vec:expr) => {
        let $array = $cx.empty_array();
        for (i, &elem) in $vec.iter().enumerate() {
            let elem = $cx.number(elem);
            $array.set($mut_cx, i as u32, elem)?;
        }
    };
}

#[macro_export]
macro_rules! create_bigint_array {
    ($cx:ident, $mut_cx:expr, $array:ident, $vec:expr) => {
        let $array = $cx.empty_array();
        for (i, &elem) in $vec.iter().enumerate() {
            let elem = JsBigInt::from_i64($mut_cx, elem);
            $array.set($mut_cx, i as u32, elem)?;
        }
    };
}

#[macro_export]
macro_rules! create_string_array {
    ($cx:ident, $mut_cx:expr, $array:ident, $vec:expr) => {
        let $array = $cx.empty_array();
        for (i, elem) in $vec.iter().enumerate() {
            let elem = $cx.string(elem);
            $array.set($mut_cx, i as u32, elem)?;
        }
    };
}
