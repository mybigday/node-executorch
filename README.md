# node-executorch

**node-executorch:** Node.js binding for ExecuTorch

# Install

```sh
npm install node-executorch
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

## Building node-executorch

Building node-executorch requires a [supported version of Node and Rust](https://github.com/neon-bindings/neon#platform-support).

To run the build, run:

```sh
$ EXECUTORCH_INSTALL_PREFIX=path/to/executorch/cmake-out yarn build
```

This command uses the [@neon-rs/cli](https://www.npmjs.com/package/@neon-rs/cli) utility to assemble the binary Node addon from the output of `cargo`.

## Available Scripts

In the project directory, you can run:

#### `yarn install`

Installs the project, including running `yarn build`.

#### `yarn build`

Builds the Node addon (`bin/<platform>/<arch>/executorch.node`) from source, generating a release build with `cargo --release`.

Additional [`cargo build`](https://doc.rust-lang.org/cargo/commands/cargo-build.html) arguments may be passed to `npm run build` and similar commands. For example, to enable a [cargo feature](https://doc.rust-lang.org/cargo/reference/features.html):

```
yarn build --feature=beetle
```

#### `yarn debug`

Similar to `yarn build` but generates a debug build with `cargo`.

#### `yarn cross`

Similar to `yarn build` but uses [cross-rs](https://github.com/cross-rs/cross) to cross-compile for another platform. Use the [`CARGO_BUILD_TARGET`](https://doc.rust-lang.org/cargo/reference/config.html#buildtarget) environment variable to select the build target.

#### `yarn test`

Runs the unit tests by calling `cargo test`. You can learn more about [adding tests to your Rust code](https://doc.rust-lang.org/book/ch11-01-writing-tests.html) from the [Rust book](https://doc.rust-lang.org/book/).

## Project Layout

The directory structure of this project is:

```
node-executorch/
├── Cargo.toml
├── README.md
├── src/
|   └── lib.rs
├── lib/
|   ├── index.ts
|   └── binding.ts
├── bin/
|   └── <platform>/<arch>/executorch.node
├── scripts/
|   └── postneon-dist.js
├── package.json
└── target/
```

| Entry          | Purpose                                                                                                                                  |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `Cargo.toml`   | The Cargo [manifest file](https://doc.rust-lang.org/cargo/reference/manifest.html), which informs the `cargo` command.                   |
| `README.md`    | This file.                                                                                                                               |
| `src/`         | The directory tree containing the Rust source code for the project.                                                                      |
| `lib.rs`       | Entry point for the Rust source code.                                                                                                    |
| `lib/`         | The directory tree containing the TypeScript source code for the project.                                                                |
| `executorch.node` | [Node addon](https://nodejs.org/api/addons.html) generated by the build.                                                              |
| `scripts/`     | Utility scripts for the project.                                                                                                         |
| `postneon-dist.js` | Utility script for post-processing the Neon build.                                                                                   |
| `package.json` | The npm [manifest file](https://docs.npmjs.com/cli/v7/configuring-npm/package-json), which informs the `npm` command.                    |
| `target/`      | Binary artifacts generated by the Rust build.                                                                                            |

## Learn More

Learn more about:

- [Neon](https://neon-bindings.com).
- [Rust](https://www.rust-lang.org).
- [Node](https://nodejs.org).

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
