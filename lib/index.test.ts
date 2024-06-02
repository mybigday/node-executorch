import path from "path";
import { Module, Tensor } from "./index";

const model = path.resolve(__dirname, "__fixtures__/mul.pte");

it("Module", async () => {
  const mod = await Module.load(model);
  expect(mod.method_names).toEqual(["forward"]);
  expect(mod.getMethodMeta("forward")).toEqual({
    name: "forward",
    inputs: [
      { tag: "tensor", tensor_info: { dtype: "float32", shape: [3, 2] } },
      { tag: "tensor", tensor_info: { dtype: "float32", shape: [3, 2] } },
    ],
    outputs: [{ tag: "tensor", tensor_info: { dtype: "float32", shape: [3, 2] } }],
  });
  { // execute without inputs
    const outputs = await mod.execute("forward");
    expect(outputs[0]).toBeInstanceOf(Tensor);
    if (outputs[0] instanceof Tensor) {
      expect(outputs[0].dtype).toBe("float32");
      expect(outputs[0].shape).toEqual([3, 2]);
      expect(outputs[0].data).toMatchSnapshot();
    }
  }
  { // forward
    const input = new Tensor("float32", [3, 2], new Float32Array([1, 2, 3, 4, 5, 6]));
    const outputs = await mod.forward([input, input]);
    expect(outputs[0]).toBeInstanceOf(Tensor);
    if (outputs[0] instanceof Tensor) {
      expect(outputs[0].dtype).toBe("float32");
      expect(outputs[0].shape).toEqual([3, 2]);
      expect(outputs[0].data).toMatchSnapshot();
    }
  }
});

it("Tensor", async () => {
  const input = new Tensor("float32", [3, 2], new Float32Array([1, 2, 3, 4, 5, 6]));

  // slice
  const slice = input.slice(null, [1, null]);
  expect(slice.dtype).toBe("float32");
  expect(slice.shape).toEqual([3, 1]);
  expect(slice.data).toMatchSnapshot();

  // setValue
  slice.setValue([0, 0], 0);
  expect(slice.data).toMatchSnapshot();

  // concat
  const concat = Tensor.concat([input, input], 1);
  expect(concat.dtype).toBe("float32");
  expect(concat.shape).toEqual([3, 4]);
  expect(concat.data).toMatchSnapshot();

  // reshape
  input.reshape([2, 3]);
  expect(input.dtype).toBe("float32");
  expect(input.shape).toEqual([2, 3]);
  expect(input.data).toMatchSnapshot();
});
