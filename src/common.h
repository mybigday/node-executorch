#pragma once

#define RETURN_IF_NOT(env, condition, message) \
  if (!(condition)) { \
    Napi::Error::New(env, message).ThrowAsJavaScriptException(); \
    return env.Undefined(); \
  }

#define THROW_IF_NOT(env, condition, message) \
  if (!(condition)) { \
    Napi::Error::New(env, message).ThrowAsJavaScriptException(); \
  }
