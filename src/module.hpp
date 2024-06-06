#include <algorithm>
#include <executorch/extension/module/module.h>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

class ModuleHolder {
  using Module = torch::executor::Module;

public:
  ModuleHolder(std::string path) {
    module_ = std::make_shared<Module>(path, Module::MlockConfig::NoMlock);
    auto method_names = module_->method_names();
    if (method_names.ok()) {
      auto names = method_names.get();
      for (auto &name : names) {
        method_names_.push_back(name);
      }
    }
  }

  ModuleHolder(const ModuleHolder &other) {
    module_ = other.module_;
    method_names_ = other.method_names_;
  }

  Module &get_module() const { return *module_; }

  const std::vector<std::string> &method_names() const { return method_names_; }

  bool has_method(const std::string &method_name) const {
    return std::find(method_names_.begin(), method_names_.end(), method_name) !=
           method_names_.end();
  }

private:
  std::shared_ptr<Module> module_ = nullptr;
  std::vector<std::string> method_names_;
};
