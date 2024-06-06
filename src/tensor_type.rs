#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TensorType {
    UInt8 = 0,
    Int8 = 1,
    Int16 = 2,
    Int32 = 3,
    Int64 = 4,
    Float16 = 5,
    Float32 = 6,
    Float64 = 7,
    ComplexFloat16 = 8,
    ComplexFloat32 = 9,
    ComplexFloat64 = 10,
    Bool = 11,
    QInt8 = 12,
    QUInt8 = 13,
    QInt32 = 14,
    BFloat16 = 15,
    QUInt4x2 = 16,
    QUInt2x4 = 17,
    Bits1x8 = 18,
    Bits2x4 = 19,
    Bits4x2 = 20,
    Bits8 = 21,
    Bits16 = 22,
}

impl From<i32> for TensorType {
    fn from(value: i32) -> Self {
        match value {
            0 => TensorType::UInt8,
            1 => TensorType::Int8,
            2 => TensorType::Int16,
            3 => TensorType::Int32,
            4 => TensorType::Int64,
            5 => TensorType::Float16,
            6 => TensorType::Float32,
            7 => TensorType::Float64,
            8 => TensorType::ComplexFloat16,
            9 => TensorType::ComplexFloat32,
            10 => TensorType::ComplexFloat64,
            11 => TensorType::Bool,
            12 => TensorType::QInt8,
            13 => TensorType::QUInt8,
            14 => TensorType::QInt32,
            15 => TensorType::BFloat16,
            16 => TensorType::QUInt4x2,
            17 => TensorType::QUInt2x4,
            18 => TensorType::Bits1x8,
            19 => TensorType::Bits2x4,
            20 => TensorType::Bits4x2,
            21 => TensorType::Bits8,
            22 => TensorType::Bits16,
            _ => panic!("Unknown tensor type"),
        }
    }
}

impl From<TensorType> for i32 {
    fn from(value: TensorType) -> Self {
        match value {
            TensorType::UInt8 => 0,
            TensorType::Int8 => 1,
            TensorType::Int16 => 2,
            TensorType::Int32 => 3,
            TensorType::Int64 => 4,
            TensorType::Float16 => 5,
            TensorType::Float32 => 6,
            TensorType::Float64 => 7,
            TensorType::ComplexFloat16 => 8,
            TensorType::ComplexFloat32 => 9,
            TensorType::ComplexFloat64 => 10,
            TensorType::Bool => 11,
            TensorType::QInt8 => 12,
            TensorType::QUInt8 => 13,
            TensorType::QInt32 => 14,
            TensorType::BFloat16 => 15,
            TensorType::QUInt4x2 => 16,
            TensorType::QUInt2x4 => 17,
            TensorType::Bits1x8 => 18,
            TensorType::Bits2x4 => 19,
            TensorType::Bits4x2 => 20,
            TensorType::Bits8 => 21,
            TensorType::Bits16 => 22,
        }
    }
}

impl From<TensorType> for usize {
    fn from(value: TensorType) -> Self {
        match value {
            TensorType::UInt8 => std::mem::size_of::<u8>(),
            TensorType::Int8 => std::mem::size_of::<i8>(),
            TensorType::Int16 => std::mem::size_of::<i16>(),
            TensorType::Int32 => std::mem::size_of::<i32>(),
            TensorType::Int64 => std::mem::size_of::<i64>(),
            TensorType::Float32 => std::mem::size_of::<f32>(),
            TensorType::Float64 => std::mem::size_of::<f64>(),
            TensorType::Bool => std::mem::size_of::<bool>(),
            _ => 0,
        }
    }
}
