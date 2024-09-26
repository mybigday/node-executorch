import type { ExternalObject, InternalEValue, MethodMeta, Optional, TensorPtrInfo, DType } from "./types";

interface Binding {
  // module methods
  moduleLoad(path: string, load_mode: number): Promise<ExternalObject>;
  moduleLoadMethod(ptr: ExternalObject, name: string): Promise<void>;
  moduleExecute(ptr: ExternalObject, method_name: string, inputs: InternalEValue[]): Promise<InternalEValue[]>;
  moduleGetMethodMeta(ptr: ExternalObject, method_name: string): MethodMeta;
  moduleMethodNames(ptr: ExternalObject): string[];
  // tensor methods
  createTensor(dtype: DType, shape: number[], data: ArrayBuffer): ExternalObject;
  tensorGetDtype(ptr: ExternalObject): DType;
  tensorGetShape(ptr: ExternalObject): number[];
  tensorGetData(ptr: ExternalObject): ArrayBuffer;
  tensorSetData(ptr: ExternalObject, data: ArrayBuffer): void;
  tensorSetValue(ptr: ExternalObject, position: Array<number>, data: number | boolean): void;
  tensorConcat(ptrs: ExternalObject[], axis: number): TensorPtrInfo;
  tensorSlice(ptr: ExternalObject, slice_position: Array<Optional<Array<Optional<number>>|number>>): TensorPtrInfo;
  tensorReshape(ptr: ExternalObject, shape: number[]): TensorPtrInfo;
}

const moduleBasePath = `../bin/${process.platform}/${process.arch}`;

if (process.platform === "win32") {
  process.env.PATH = `${moduleBasePath};${process.env.PATH}`;
}

export const mod = require(`${moduleBasePath}/executorch.node`) as Binding;
