import { mod } from "./binding";
import { EValueTag } from "./types";
import type { ExternalObject, MethodMeta, TensorData, DType, Optional, TensorPtrInfo, InternalEValue } from "./types";

export * from "./types";

class Tensor {
  ptr: ExternalObject;
  shape: number[];
  dtype: DType;

  constructor(dtype: DType, shape: number[], data: TensorData | ExternalObject) {
    this.shape = shape;
    this.dtype = dtype;

    if (!Array.isArray(data) && !ArrayBuffer.isView(data)) {
      this.ptr = data;
      return;
    }

    switch (dtype) {
      case "uint8":
        this.ptr = mod.createU8Tensor(data as Uint8Array, shape);
        break;
      case "int8":
        this.ptr = mod.createI8Tensor(data as Int8Array, shape);
        break;
      case "int16":
        this.ptr = mod.createI16Tensor(data as Int16Array, shape);
        break;
      case "int32":
        this.ptr = mod.createI32Tensor(data as Int32Array, shape);
        break;
      case "int64":
        this.ptr = mod.createI64Tensor(data as BigInt64Array, shape);
        break;
      case "float32":
        this.ptr = mod.createF32Tensor(data as Float32Array, shape);
        break;
      case "float64":
        this.ptr = mod.createF64Tensor(data as Float64Array, shape);
        break;
      default:
        throw new Error(`Unsupported dtype: ${dtype}`);
    }
  }

  static fromPtr(ptrInfo: TensorPtrInfo): Tensor {
    return new Tensor(ptrInfo.dtype, ptrInfo.shape, ptrInfo.ptr);
  }

  static concat(tensors: Tensor[], axis: number): Tensor {
    const ptrs = tensors.map((t) => t.ptr);
    const ptrInfo = mod.tensorConcat(ptrs, axis);
    return new Tensor(ptrInfo.dtype, ptrInfo.shape, ptrInfo.ptr);
  }

  slice(...slice_position: Array<Optional<Array<Optional<number>>|number>>): Tensor {
    const ptrInfo = mod.tensorSlice(this.ptr, ...slice_position);
    return new Tensor(ptrInfo.dtype, ptrInfo.shape, ptrInfo.ptr);
  }

  reshape(shape: number[]): Tensor {
    mod.tensorReshape(this.ptr, shape);
    return this;
  }

  get data(): TensorData {
    return mod.tensorGetData(this.ptr);
  }

  set data(data: TensorData) {
    mod.tensorSetData(this.ptr, data);
  }

  setValue(position: Array<number>, data: number | boolean): void {
    mod.tensorSetValue(this.ptr, position, data);
  }

  dispose() {
    mod.tensorDispose(this.ptr);
  }
}

export type EValue = null | string | number | boolean | Tensor | undefined;

const toInternalEValue = (value: EValue): InternalEValue => {
  if (value === null) {
    return { tag: EValueTag.Null, data: null };
  } else if (typeof value === "string") {
    return { tag: EValueTag.String, data: value };
  } else if (typeof value === "number") {
    if (Number.isInteger(value)) {
      return { tag: EValueTag.Int, data: value };
    } else {
      return { tag: EValueTag.Double, data: value };
    }
  } else if (typeof value === "boolean") {
    return { tag: EValueTag.Bool, data: value };
  } else if (value instanceof Tensor) {
    return {
      tag: EValueTag.Tensor,
      data: { dtype: value.dtype, shape: value.shape, ptr: value.ptr },
    };
  } else {
    throw new Error(`Unsupported type: ${typeof value}`);
  }
}

const fromInternalEValue = (value: InternalEValue): EValue => {
  switch (value.tag) {
    case EValueTag.Null:
      return null;
    case EValueTag.String:
    case EValueTag.Int:
    case EValueTag.Double:
    case EValueTag.Bool:
      return value.data as string | number | boolean;
    case EValueTag.Tensor:
      return Tensor.fromPtr(value.data as TensorPtrInfo);
    default:
      return undefined;
  }
}

class Module {
  ptr: ExternalObject;

  constructor(ptr: ExternalObject) {
    this.ptr = ptr;
  }

  static async load(path: string): Promise<Module> {
    const ptr = await mod.moduleLoad(path);
    return new Module(ptr);
  }

  get method_names(): string[] {
    return mod.moduleMethodNames(this.ptr);
  }

  getMethodMeta(method_name: string): MethodMeta {
    return mod.moduleGetMethodMeta(this.ptr, method_name);
  }

  async loadMethod(name: string): Promise<void> {
    await mod.moduleLoadMethod(this.ptr, name);
  }

  async forward(inputs: EValue[]): Promise<EValue[]> {
    return this.execute("forward", inputs);
  }

  async execute(method_name: string, inputs: EValue[] = []): Promise<EValue[]> {
    return (
      await mod.moduleExecute(
        this.ptr,
        method_name,
        inputs.map(toInternalEValue)
      )
    ).map(fromInternalEValue);
  }
}

class Sampler {
  ptr: ExternalObject;
  vocab_size: number;

  constructor(vocab_size: number, temperature: number = 0.7, topP: number = 0.9, seed?: number) {
    this.vocab_size = vocab_size;
    this.ptr = mod.createSampler(
      vocab_size,
      temperature,
      topP,
      seed ?? Math.floor(Math.random() * 1000000)
    );
  }

  sample(tensor: Tensor): number {
    if (tensor.dtype !== "float32") {
      throw new Error(`Unsupported dtype: ${tensor.dtype}`);
    }
    if (tensor.shape.length !== 3 || tensor.shape[0] !== 1 || tensor.shape[1] === 0 || tensor.shape[2] !== this.vocab_size) {
      throw new Error(`Invalid shape: ${tensor.shape}`);
    }
    if (tensor.shape[1] > 1) {
      const data = tensor.data as Float32Array;
      return mod.samplerSample(this.ptr, data.subarray(data.length - this.vocab_size));
    } else {
      return mod.samplerSample(this.ptr, tensor.data as Float32Array);
    }
  }
}

export { Sampler, Tensor, Module };
