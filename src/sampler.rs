use neon::prelude::*;
use cpp::{cpp, cpp_class};

cpp! {{
  #include <vector>
  #include <executorch/examples/models/llama2/sampler/sampler.h>
  #include <executorch/examples/models/llama2/sampler/sampler.cpp>
}}

cpp_class!(pub unsafe struct Sampler as "torch::executor::Sampler");

impl Sampler {
  pub fn new(vocab_size: i32, temperature: f32, topp: f32, rng_seed: u64) -> Self {
      unsafe {
          cpp!([vocab_size as "int", temperature as "float", topp as "float", rng_seed as "uint64_t"] -> Sampler as "torch::executor::Sampler" {
              return torch::executor::Sampler(vocab_size, temperature, topp, rng_seed);
          })
      }
  }

  pub fn sample(&self, param : Vec<f32>) -> i32 {
      unsafe {
          cpp!([self as "torch::executor::Sampler*", param as "std::vector<float>"] -> i32 as "int32_t" {
              auto data = new float[param.size()];
              memcpy(data, param.data(), param.size() * sizeof(float));
              auto result = self->sample(data);
              delete[] data;
              return result;
          })
      }
  }
}

impl Finalize for Sampler {}
