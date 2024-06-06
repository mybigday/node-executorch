#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ETError {
    Ok = 0,
    Internal = 1,
    InvalidState = 2,
    EndOfMethod = 3,
    NotSupported = 16,
    NotImplemented = 17,
    InvalidArgument = 18,
    InvalidType = 19,
    OperatorMissing = 20,
    NotFound = 32,
    MemoryAllocationFailed = 33,
    AccessFailed = 34,
    InvalidProgram = 35,
    DelegateInvalidCompatibility = 48,
    DelegateMemoryAllocationFailed = 49,
    DelegateInvalidHandle = 50,
}

impl From<i32> for ETError {
    fn from(value: i32) -> Self {
        match value {
            0 => ETError::Ok,
            1 => ETError::Internal,
            2 => ETError::InvalidState,
            3 => ETError::EndOfMethod,
            16 => ETError::NotSupported,
            17 => ETError::NotImplemented,
            18 => ETError::InvalidArgument,
            19 => ETError::InvalidType,
            20 => ETError::OperatorMissing,
            32 => ETError::NotFound,
            33 => ETError::MemoryAllocationFailed,
            34 => ETError::AccessFailed,
            35 => ETError::InvalidProgram,
            48 => ETError::DelegateInvalidCompatibility,
            49 => ETError::DelegateMemoryAllocationFailed,
            50 => ETError::DelegateInvalidHandle,
            _ => panic!("Unknown error code"),
        }
    }
}

impl From<ETError> for i32 {
    fn from(value: ETError) -> Self {
        match value {
            ETError::Ok => 0,
            ETError::Internal => 1,
            ETError::InvalidState => 2,
            ETError::EndOfMethod => 3,
            ETError::NotSupported => 16,
            ETError::NotImplemented => 17,
            ETError::InvalidArgument => 18,
            ETError::InvalidType => 19,
            ETError::OperatorMissing => 20,
            ETError::NotFound => 32,
            ETError::MemoryAllocationFailed => 33,
            ETError::AccessFailed => 34,
            ETError::InvalidProgram => 35,
            ETError::DelegateInvalidCompatibility => 48,
            ETError::DelegateMemoryAllocationFailed => 49,
            ETError::DelegateInvalidHandle => 50,
        }
    }
}
