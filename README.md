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

Note: For Windows currently only support cross-compile.

1. Fetch ExecuTorch Source
2. Build ExecuTorch and install to any path
3. Build this project

```sh
# Install dependency
yarn

# Build
yarn build --CDCMAKE_PREFIX_PATH=/path/to/install/dir \
  --CDEXECUTORCH_SRC_ROOT=/path/to/executorch/src_root
```

## License

BSD

---

<p align="center">
  <a href="https://bricks.tools">
    <img width="90px" src="https://avatars.githubusercontent.com/u/17320237?s=200&v=4">
  </a>
  <p align="center">
    Built and maintained by <a href="https://bricks.tools">BRICKS</a>.
  </p>
</p>
