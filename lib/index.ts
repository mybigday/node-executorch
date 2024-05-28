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

interface TensorImpl {
  get dtype(): DType;
  get shape(): number[];
  get data(): TensorData;
}

interface Tensor {
  new (dtype: DType, shape: number[], data: TensorData): TensorImpl;
}

type EValue = null | string | number | boolean | TensorImpl;

interface ModuleImpl {
  forward(inputs: EValue[]): Promise<EValue[]>;
  execute(method_name: string, inputs: EValue[]): Promise<EValue[]>;
  get method_names(): string[];
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
