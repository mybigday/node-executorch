type DType =
  | "float32"
  | "float64"
  | "int32"
  | "uint8"
  | "int8"
  | "int16"
  | "int64"
  | "bool";

type TensorData =
  | boolean[]
  | Float32Array
  | Float64Array
  | Int32Array
  | Uint8Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigInt64Array;

type Optional<T> = T | null | undefined;

interface TensorImpl {
  get dtype(): DType;
  get shape(): number[];
  get data(): TensorData;
  setIndex(position: Array<number>, data: number | boolean): void
  slice(slice_position: Array<Optional<Optional<number>[]>>): TensorImpl;
  reshape(shape: number[]): TensorImpl;
  dispose(): void;
}

interface Tensor {
  new(dtype: DType, shape: number[], data: TensorData): TensorImpl;
  concat(tensors: TensorImpl[], axis: number): TensorImpl;
}

type EValue = null | string | number | boolean | TensorImpl;

type TensorInfo = {
  dtype: DType;
  shape: number[];
};

type EValueSpec = {
  tag: "null" | "string" | "number" | "boolean" | "tensor";
  tensor_info?: TensorInfo;
};

type MethodMeta = {
  name: string;
  inputs: Array<EValueSpec | undefined>;
  outputs: Array<EValueSpec | undefined>;
};

interface ModuleImpl {
  forward(inputs: EValue[]): Promise<EValue[]> | undefined;
  execute(method_name: string, inputs?: EValue[]): Promise<EValue[]> | undefined;
  getMethodMeta(method_name: string): MethodMeta | undefined;
  get method_names(): string[];
  dispose(): void;
}

interface Module {
  load(path: string): Promise<ModuleImpl>;
}

interface Binding {
  Module: Module;
  Tensor: Tensor;
}

const mod = require(
  `../bin/${process.platform}/${process.arch}/node-executorch.node`,
) as Binding;

export const Module = mod.Module;
export const Tensor = mod.Tensor;
