import path from "path";
import { Module, Tensor, BPETokenizer } from "./index";

const modelPath = path.resolve(__dirname, "__fixtures__/mul.pte");

const tokenizerPath = path.resolve(__dirname, "__fixtures__/tokenizer.model");

it("Module", async () => {
  const mod = await Module.load(modelPath);
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

it("Tensor", async () => {
  const input = new Tensor("float32", [3, 2], new Float32Array([1, 2, 3, 4, 5, 6]));
  const slice = input.slice([null, [1, null]]);
  expect(slice.dtype).toBe("float32");
  expect(slice.shape).toEqual([3, 1]);
  expect(slice.data).toMatchSnapshot();

  slice.setIndex([0, 0], 0);
  expect(slice.data).toMatchSnapshot();

  const concat = Tensor.concat([input, input], 1);
  expect(concat.dtype).toBe("float32");
  expect(concat.shape).toEqual([3, 4]);
  expect(concat.data).toMatchSnapshot();
});


it("Tokenizer", async () => {
  const tokenizer = await BPETokenizer.load(tokenizerPath);
  expect(tokenizer.vocab_size).toBe(32000);
  expect(tokenizer.bos_token_id).toBe(1n);
  expect(tokenizer.eos_token_id).toBe(2n);
  expect(tokenizer.encode("Hello, world!", 1)).toMatchSnapshot();
  expect(tokenizer.decode(3n, 0n)).toBe("Hello");
});
