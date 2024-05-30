#include "Tokenizer.h"
#include "utils.h"
#include <string>

namespace executorch::node {

#define CHECK_DISPOSED(env)                                                                                                \
  if (!tokenizer_) {                                                                                                    \
    Napi::TypeError::New(env, "Tokenizer is disposed").ThrowAsJavaScriptException();                                    \
    return env.Undefined();                                                                                             \
  }

/* LoadWorker */

template <typename TokenizerCls>
class LoadWorker : public Napi::AsyncWorker, public Napi::Promise::Deferred {
public:
  LoadWorker(Napi::Env env, const std::string &path, int64_t vocab_size, int64_t bos_tok, int64_t eos_tok)
      : Napi::AsyncWorker(env),
        Napi::Promise::Deferred(env),
        path_(path),
        vocab_size_(vocab_size),
        bos_tok_(bos_tok),
        eos_tok_(eos_tok) {}

protected:
  void Execute() {
    try {
      tokenizer_ = std::make_unique<TokenizerCls>(vocab_size_, bos_tok_, eos_tok_);
      auto error = tokenizer_->load(path_);
      if (error != torch::executor::Error::Ok) {
        throw std::runtime_error("Failed to load tokenizer: " + errorString(error));
      }
    } catch (const std::exception &e) {
      SetError(e.what());
    }
  }

  void OnOK() {
    Resolve(Tokenizer<TokenizerCls>::New(tokenizer_.release()));
  }

  void OnError(const Napi::Error &e) { Reject(e.Value()); }

private:
  const std::string path_;
  std::unique_ptr<TokenizerCls> tokenizer_;
  int64_t vocab_size_;
  int64_t bos_tok_;
  int64_t eos_tok_;
};

/* Tokenizer */

// static load(path: string, vocab_size: number = 32000,
//             bos_tok: number = 1, eos_tok: number = 2): Promise<Tokenizer>
template <typename TokenizerCls>
Napi::Value Tokenizer<TokenizerCls>::Load(const Napi::CallbackInfo &info) {
  auto env = info.Env();
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "Expected a string").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (info.Length() > 1 && !info[1].IsNumber()) {
    Napi::TypeError::New(env, "Expected a number").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (info.Length() > 2 && !info[2].IsNumber()) {
    Napi::TypeError::New(env, "Expected a number").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (info.Length() > 3 && !info[3].IsNumber()) {
    Napi::TypeError::New(env, "Expected a number").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  bool *lossless;

  auto path = info[0].As<Napi::String>().Utf8Value();
  int64_t vocab_size = info.Length() > 1 ? info[1].As<Napi::BigInt>().Int64Value(lossless) : 32000;
  int64_t bos_tok = info.Length() > 2 ? info[2].As<Napi::BigInt>().Int64Value(lossless) : 1;
  int64_t eos_tok = info.Length() > 3 ? info[3].As<Napi::BigInt>().Int64Value(lossless) : 2;
  auto worker = new LoadWorker<TokenizerCls>(env, path, vocab_size, bos_tok, eos_tok);
  worker->Queue();
  return worker->Promise();
}

// get vocab_size(): number
template <typename TokenizerCls>
Napi::Value Tokenizer<TokenizerCls>::GetVocabSize(const Napi::CallbackInfo &info) {
  auto env = info.Env();
  CHECK_DISPOSED(env);
  return Napi::Number::New(env, tokenizer_->vocab_size());
}

// get bos_token(): number
template <typename TokenizerCls>
Napi::Value Tokenizer<TokenizerCls>::GetBosToken(const Napi::CallbackInfo &info) {
  auto env = info.Env();
  if (!tokenizer_) {
    Napi::TypeError::New(env, "Tokenizer is disposed").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  return Napi::BigInt::New(env, tokenizer_->bos_tok());
}

// get eos_token(): number
template <typename TokenizerCls>
Napi::Value Tokenizer<TokenizerCls>::GetEosToken(const Napi::CallbackInfo &info) {
  auto env = info.Env();
  CHECK_DISPOSED(env);
  return Napi::BigInt::New(env, tokenizer_->eos_tok());
}

// encode(text: string, prepend_bos: number = 1, append_eos: number = 0): BigUint64Array
template <typename TokenizerCls>
Napi::Value Tokenizer<TokenizerCls>::Encode(const Napi::CallbackInfo &info) {
  auto env = info.Env();
  CHECK_DISPOSED(env);
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "Expected a string").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (info.Length() > 1 && !info[1].IsNumber()) {
    Napi::TypeError::New(env, "Expected a number").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (info.Length() > 2 && !info[2].IsNumber()) {
    Napi::TypeError::New(env, "Expected a number").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  auto text = info[0].As<Napi::String>().Utf8Value();
  int8_t prepend_bos = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 1;
  int8_t append_eos = info.Length() > 2 ? info[2].As<Napi::Number>().Int32Value() : 0;
  fprintf(stderr, "encode: %s\n", text.c_str());
  fflush(stderr);
  auto result = tokenizer_->encode(text, prepend_bos, append_eos);
  if (!result.ok()) {
    Napi::TypeError::New(env, "Failed to encode text: " + errorString(result.error())).ThrowAsJavaScriptException();
    return env.Undefined();
  }
  fprintf(stderr, "encode: %d\n", result.get().size());
  fflush(stderr);
  auto ids = result.get();
  auto arr = Napi::BigUint64Array::New(env, ids.size());
  memcpy(arr.Data(), ids.data(), ids.size() * sizeof(uint64_t));
  return arr;
}

// decode(prev: BigInt, token: BigInt): string
template <typename TokenizerCls>
Napi::Value Tokenizer<TokenizerCls>::Decode(const Napi::CallbackInfo &info) {
  auto env = info.Env();
  CHECK_DISPOSED(env);
  if (info.Length() < 2 || !info[0].IsBigInt() || !info[1].IsBigInt()) {
    Napi::TypeError::New(env, "Expected two bigints").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  bool *lossless;

  auto prev = info[0].As<Napi::BigInt>().Uint64Value(lossless);
  auto token = info[1].As<Napi::BigInt>().Uint64Value(lossless);
  auto result = tokenizer_->decode(prev, token);
  if (!result.ok()) {
    Napi::TypeError::New(env, "Failed to decode token").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  return Napi::String::New(env, result.get());
}

// dispose(): void
template <typename TokenizerCls>
void Tokenizer<TokenizerCls>::Dispose(const Napi::CallbackInfo &info) {
  tokenizer_.reset();
}

template <typename TokenizerCls>
Napi::Object Tokenizer<TokenizerCls>::Init(Napi::Env env, Napi::Object exports) {
  using Class = Tokenizer<TokenizerCls>;
  using ObjectWrap = Napi::ObjectWrap<Class>;
  Napi::Function func =
      ObjectWrap::DefineClass(env, name(),
                  {ObjectWrap::StaticMethod("load", &Class::Load),
                   ObjectWrap::InstanceAccessor("vocab_size", &Class::GetVocabSize, nullptr),
                   ObjectWrap::InstanceAccessor("bos_token_id", &Class::GetBosToken, nullptr),
                   ObjectWrap::InstanceAccessor("eos_token_id", &Class::GetEosToken, nullptr),
                   ObjectWrap::InstanceMethod("encode", &Class::Encode),
                   ObjectWrap::InstanceMethod("decode", &Class::Decode),
                   ObjectWrap::InstanceMethod("dispose", &Class::Dispose)});

  constructor = Napi::Persistent(func);
  constructor.SuppressDestruct();
  exports.Set(name(), func);

  return exports;
}

template class LoadWorker<torch::executor::Tiktoken>;
template class LoadWorker<torch::executor::BPETokenizer>;

template<typename TokenizerCls> Napi::FunctionReference Tokenizer<TokenizerCls>::constructor;

template class Tokenizer<torch::executor::Tiktoken>;
template class Tokenizer<torch::executor::BPETokenizer>;

} // namespace executorch::node
