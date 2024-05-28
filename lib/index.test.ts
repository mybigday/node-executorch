import path from "path";
import { Module, Tensor } from "./index";

const model = path.resolve(__dirname, "__fixtures__/mul.pte");

it("work fine", async () => {
  const mod = await Module.load(model);
  expect(mod.method_names).toEqual(["forward"]);
  const input = new Tensor("float32", [3, 2], new Float32Array([1, 2, 3, 4, 5, 6]));
  const outputs = await mod.forward([input, input]);
  expect(outputs[0]).toBeInstanceOf(Tensor);
  if (outputs[0] instanceof Tensor) {
    expect(outputs[0].dtype).toBe("float32");
    expect(outputs[0].shape).toEqual([3, 2]);
    expect(outputs[0].data).toMatchSnapshot();
  }
});
