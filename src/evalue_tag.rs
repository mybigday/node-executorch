#[derive(Clone, PartialEq)]
pub enum EValueTag {
    None = 0,
    Tensor = 1,
    String = 2,
    Double = 3,
    Int = 4,
    Bool = 5,
    ListBool = 6,
    ListDouble = 7,
    ListInt = 8,
    ListTensor = 9,
    ListScalar = 10,
    ListOptionalTensor = 11,
}

impl From<i32> for EValueTag {
    fn from(value: i32) -> Self {
        match value {
            0 => EValueTag::None,
            1 => EValueTag::Tensor,
            2 => EValueTag::String,
            3 => EValueTag::Double,
            4 => EValueTag::Int,
            5 => EValueTag::Bool,
            6 => EValueTag::ListBool,
            7 => EValueTag::ListDouble,
            8 => EValueTag::ListInt,
            9 => EValueTag::ListTensor,
            10 => EValueTag::ListScalar,
            11 => EValueTag::ListOptionalTensor,
            _ => panic!("Unknown EValueTag"),
        }
    }
}

impl From<EValueTag> for i32 {
    fn from(value: EValueTag) -> Self {
        match value {
            EValueTag::None => 0,
            EValueTag::Tensor => 1,
            EValueTag::String => 2,
            EValueTag::Double => 3,
            EValueTag::Int => 4,
            EValueTag::Bool => 5,
            EValueTag::ListBool => 6,
            EValueTag::ListDouble => 7,
            EValueTag::ListInt => 8,
            EValueTag::ListTensor => 9,
            EValueTag::ListScalar => 10,
            EValueTag::ListOptionalTensor => 11,
        }
    }
}
