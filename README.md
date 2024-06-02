executorch-node
===

Node.js binding for ExecuTorch

# Installation

```sh
npm i node-executorch
```

# Usage

```js
import { Module, Tensor } from 'node-executorch';

const model = await Module.load('path/to/model.pte');

const input = new Tensor('int32', [1, 1], Int32Array.from([ 1 ]));

const outputs = await model.forward([input]);

// Manually release
input.dispose();
model.dispose();
```

# Build From Source

Requirements:

- clang
- w32-mingw (This project only support cross compile windows)


```sh
```

## License

3-Clause BSD

---

<p align="center">
  <a href="https://bricks.tools">
    <img width="90px" src="https://avatars.githubusercontent.com/u/17320237?s=200&v=4">
  </a>
  <p align="center">
    Built and maintained by <a href="https://bricks.tools">BRICKS</a>.
  </p>
</p>
