import type { ExternalObject, InternalEValue, MethodMeta, TensorData, Optional, TensorPtrInfo } from "./types";

interface Binding {
  // module methods
  moduleLoad(path: string): Promise<ExternalObject>;
  moduleLoadMethod(ptr: ExternalObject, name: string): Promise<void>;
  moduleExecute(ptr: ExternalObject, method_name: string, inputs: InternalEValue[]): Promise<InternalEValue[]>;
  moduleGetMethodMeta(ptr: ExternalObject, method_name: string): MethodMeta;
  moduleMethodNames(ptr: ExternalObject): string[];
  moduleDispose(ptr: ExternalObject): void;
  // sampler methods
  createSampler(vocab_size: number, temperature: number, top_p: number, seed: number): ExternalObject;
  samplerSample(ptr: ExternalObject, vector: Float32Array): number;
  // tensor methods
  createU8Tensor(data: Uint8Array, shape: number[]): ExternalObject;
  createI8Tensor(data: Int8Array, shape: number[]): ExternalObject;
  createI16Tensor(data: Int16Array, shape: number[]): ExternalObject;
  createI32Tensor(data: Int32Array, shape: number[]): ExternalObject;
  createI64Tensor(data: BigInt64Array, shape: number[]): ExternalObject;
  createF32Tensor(data: Float32Array, shape: number[]): ExternalObject;
  createF64Tensor(data: Float64Array, shape: number[]): ExternalObject;
  tensorGetData(ptr: ExternalObject): TensorData;
  tensorSetData(ptr: ExternalObject, data: TensorData): void;
  tensorConcat(ptrs: ExternalObject[], axis: number): TensorPtrInfo;
  tensorSlice(ptr: ExternalObject, ...slice_position: Array<Optional<Array<Optional<number>>|number>>): TensorPtrInfo;
  tensorReshape(ptr: ExternalObject, shape: number[]): void;
  tensorSetValue(ptr: ExternalObject, position: Array<number>, data: number | boolean): void;
  tensorDispose(ptr: ExternalObject): void;
}

const moduleBasePath = `../bin/${process.platform}/${process.arch}`;

if (process.platform === "win32") {
  process.env.PATH = `${moduleBasePath};${process.env.PATH}`;
}

export const mod = require(`${moduleBasePath}/executorch.node`) as Binding;
