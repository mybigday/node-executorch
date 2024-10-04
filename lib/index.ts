import { mod } from "./binding";
import { EValueTag, DType, ModuleLoadMode } from "./types";
import type { ExternalObject, MethodMeta, TensorData, Optional, TensorPtrInfo, InternalEValue } from "./types";

export * from "./types";

export type DTypeStr = keyof typeof DType;

const dtypeTypedArrayMap = {
  [DType.float32]: Float32Array,
  [DType.float64]: Float64Array,
  [DType.uint8]: Uint8Array,
  [DType.int8]: Int8Array,
  [DType.int16]: Int16Array,
  [DType.int32]: Int32Array,
  [DType.int64]: BigInt64Array,
};

const toArrayBuffer = (data: TensorData): ArrayBuffer => {
  if (Array.isArray(data)) {
    return Uint8Array.from(data.map((v) => v ? 1 : 0)).buffer;
  } else {
    return data.buffer;
  }
}

class Tensor {
  _ptr: ExternalObject;
  _dtype: DType;

  constructor(dtype: DTypeStr | DType, shape: number[], data: TensorData | ExternalObject) {
    this._dtype = typeof dtype === "string" ? DType[dtype] : dtype;
    if (!Array.isArray(data) && !ArrayBuffer.isView(data)) {
      this._ptr = data;
      return;
    }
    this._ptr = mod.createTensor(this._dtype, shape, toArrayBuffer(data as TensorData));
  }

  static fromPtr(ptrInfo: TensorPtrInfo): Tensor {
    const { shape, dtype, ptr } = ptrInfo;

    return new Tensor(dtype, shape, ptr);
  }

  static concat(tensors: Tensor[], axis: number): Tensor {
    const ptrs = tensors.map((t) => t._ptr);
    return Tensor.fromPtr(mod.tensorConcat(ptrs, axis));
  }

  get shape(): number[] {
    return mod.tensorGetShape(this._ptr);
  }

  get dtype(): DTypeStr {
    return DType[this._dtype] as DTypeStr;
  }

  get data(): TensorData {
    const buf = mod.tensorGetData(this._ptr);
    if (this._dtype === DType.bool) {
      return Array.from(new Uint8Array(buf)).map((v) => v === 1);
    } else {
      return new dtypeTypedArrayMap[this._dtype](buf);
    }
  }

  set data(data: TensorData) {
    mod.tensorSetData(this._ptr, toArrayBuffer(data));
  }

  slice(...slice_position: Array<Optional<Array<Optional<number>>|number>>): Tensor {
    return Tensor.fromPtr(mod.tensorSlice(this._ptr, slice_position));
  }

  reshape(shape: number[]): Tensor {
    return Tensor.fromPtr(mod.tensorReshape(this._ptr, shape));
  }  

  setValue(position: Array<number>, data: number | boolean): void {
    mod.tensorSetValue(this._ptr, position, data);
  }

  dispose() {
    delete this._ptr;
  }
}

export type EValue = null | string | number | boolean | Tensor | undefined;

const toInternalEValue = (value: EValue): InternalEValue => {
  if (value === null) {
    return { tag: EValueTag.None, data: null };
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
      data: value._ptr,
    };
  } else {
    throw new Error(`Unsupported type: ${typeof value}`);
  }
}

const fromInternalEValue = (value: InternalEValue): EValue => {
  switch (value.tag) {
    case EValueTag.None:
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
  _ptr: ExternalObject;

  constructor(ptr: ExternalObject) {
    this._ptr = ptr;
  }

  static async load(path: string, load_mode: ModuleLoadMode = ModuleLoadMode.Mmap): Promise<Module> {
    const ptr = await mod.moduleLoad(path, load_mode);
    return new Module(ptr);
  }

  get method_names(): string[] {
    return mod.moduleMethodNames(this._ptr);
  }

  getMethodMeta(method_name: string): MethodMeta {
    return mod.moduleGetMethodMeta(this._ptr, method_name);
  }

  async loadMethod(name: string): Promise<void> {
    await mod.moduleLoadMethod(this._ptr, name);
  }

  async forward(inputs: EValue[]): Promise<EValue[]> {
    return this.execute("forward", inputs);
  }

  async execute(method_name: string, inputs: EValue[] = []): Promise<EValue[]> {
    return (
      await mod.moduleExecute(
        this._ptr,
        method_name,
        inputs.map(toInternalEValue)
      )
    ).map(fromInternalEValue);
  }

  dispose() {
    delete this._ptr;
  }
}

class Sampler {
  _ptr: ExternalObject;
  _vocab_size: number;

  constructor(vocab_size: number, temperature: number = 0.7, topP: number = 0.9, seed?: number) {
    this._vocab_size = vocab_size;
    this._ptr = mod.createSampler(
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
    if (tensor.shape.length !== 3 || tensor.shape[0] !== 1 || tensor.shape[1] === 0 || tensor.shape[2] !== this._vocab_size) {
      throw new Error(`Invalid shape: ${tensor.shape}`);
    }
    return mod.samplerSample(this._ptr, tensor._ptr);
  }

  dispose() {
    delete this._ptr;
  }
}

export { Sampler, Tensor, Module };
