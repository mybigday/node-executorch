#include <executorch/runtime/core/tag.h>
#include <unordered_map>

using Tag = torch::executor::Tag;

const std::unordered_map<Tag, int32_t> TAG_ID_MAP = {
    {Tag::None, 0},       {Tag::Tensor, 1},      {Tag::String, 2},
    {Tag::Double, 3},     {Tag::Int, 4},         {Tag::Bool, 5},
    {Tag::ListBool, 6},   {Tag::ListDouble, 7},  {Tag::ListInt, 8},
    {Tag::ListTensor, 9}, {Tag::ListScalar, 10}, {Tag::ListOptionalTensor, 11},
};

inline int32_t tag_to_int(Tag tag) { return TAG_ID_MAP.at(tag); }
