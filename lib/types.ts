export enum DType {
  uint8 = 0,
  int8 = 1,
  int16 = 2,
  int32 = 3,
  int64 = 4,
  float16 = 5,
  float32 = 6,
  float64 = 7,
  bool = 11,
}

export type TensorData =
  | boolean[]
  | Float32Array
  | Float64Array
  | Int32Array
  | Uint8Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigInt64Array;

export type Optional<T> = T | null | undefined;

export type ExternalObject = any;

export type TensorPtrInfo = {
  shape: number[];
  dtype: DType;
  ptr: ExternalObject;
};

export enum EValueTag {
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

export type InternalEValue = {
  tag: EValueTag;
  data: null | string | number | boolean | ExternalObject | TensorPtrInfo | undefined;
}

export type TensorInfo = {
  dtype: DType;
  shape: number[];
};

export type EValueSpec = {
  tag: EValueTag;
  tensor_info?: TensorInfo;
};

export type MethodMeta = {
  name: string;
  inputs: Array<EValueSpec | undefined>;
  outputs: Array<EValueSpec | undefined>;
};
