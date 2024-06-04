#[macro_export]
macro_rules! get_arg_as_vec {
  ($cx:ident, $index:expr, $js_type:ty, $type:ty) => {
    $cx.argument::<JsArray>($index)?
      .to_vec(&mut $cx)?
      .iter()
      .map(|value| value.downcast::<$js_type, _>(&mut $cx).unwrap().value(&mut $cx) as $type)
      .collect()
  };
}
