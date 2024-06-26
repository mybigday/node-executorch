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
  setValue(position: Array<number>, data: number | boolean): void
  slice(...slice_position: Array<Optional<Array<Optional<number>>|number>>): TensorImpl;
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

interface SamplerImpl {
  sample(tensor: TensorImpl): number;
  dispose(): void;
}

interface Sampler {
  new(vocab_size: number, temperature?: number, top_p?: number, seed?: number): SamplerImpl;
}

interface Binding {
  Module: Module;
  Tensor: Tensor;
  Sampler: Sampler;
}

const moduleBasePath = `../bin/${process.platform}/${process.arch}`;

if (process.platform === "linux") {
  process.env.LD_LIBRARY_PATH = `${moduleBasePath}:${process.env.LD_LIBRARY_PATH}`;
} else if (process.platform === "darwin") {
  process.env.DYLD_LIBRARY_PATH = `${moduleBasePath}:${process.env.DYLD_LIBRARY_PATH}`;
} else if (process.platform === "win32") {
  process.env.PATH = `${moduleBasePath};${process.env.PATH}`;
}

const mod = require(`${moduleBasePath}/node-executorch.node`) as Binding;

export const Module = mod.Module;
export const Tensor = mod.Tensor;
export const Sampler = mod.Sampler;
