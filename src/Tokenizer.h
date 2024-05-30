#pragma once

#include "tiktoken.h"
#include "bpe_tokenizer.h"
#include <memory>
#include <napi.h>

namespace executorch {
namespace node {

template <typename TokenizerCls>
class Tokenizer : public Napi::ObjectWrap<Tokenizer<TokenizerCls>> {
public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  Tokenizer(const Napi::CallbackInfo &info) : Napi::ObjectWrap<Tokenizer<TokenizerCls>>(info) {}

  static Napi::Object New(TokenizerCls *tokenizer) {
    auto instance = constructor.New({});
    auto *obj = Napi::ObjectWrap<Tokenizer<TokenizerCls>>::Unwrap(instance);
    obj->tokenizer_.reset(tokenizer);
    return instance;
  }

protected:
  static Napi::Value Load(const Napi::CallbackInfo &info);
  Napi::Value GetVocabSize(const Napi::CallbackInfo &info);
  Napi::Value GetBosToken(const Napi::CallbackInfo &info);
  Napi::Value GetEosToken(const Napi::CallbackInfo &info);
  Napi::Value Encode(const Napi::CallbackInfo &info);
  Napi::Value Decode(const Napi::CallbackInfo &info);
  void Dispose(const Napi::CallbackInfo &info);

  static constexpr const char* name() {
    if constexpr (std::is_same_v<TokenizerCls, torch::executor::Tiktoken>) {
      return "Tiktoken";
    } else if constexpr (std::is_same_v<TokenizerCls, torch::executor::BPETokenizer>) {
      return "BPETokenizer";
    }
  }

private:
  static Napi::FunctionReference constructor;
  std::unique_ptr<TokenizerCls> tokenizer_ = nullptr;
};

typedef Tokenizer<torch::executor::Tiktoken> Tiktoken;
typedef Tokenizer<torch::executor::BPETokenizer> BPETokenizer;

} // namespace node
} // namespace executorch
