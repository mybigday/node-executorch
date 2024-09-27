const fs = require('fs');
const path = require('path');

let platform = 'unknown';
let arch = 'unknown';

// Parse file header to determine platform and architecture
const content = fs.readFileSync('index.node').subarray(0, 256);

if (content[0] === 0x7f && content[1] === 0x45 && content[2] === 0x4c && content[3] === 0x46) { // Linux: ELF x86_64/aarch64
  platform = 'linux';
  if (content[18] === 0xb7) {
    arch = 'arm64';
  } else if (content[18] === 0x3E) {
    arch = 'x64';
  } else {
    console.error('Unknown ELF arch code:', content[18].toString(16));
  }
} else if (content[0] === 0x4d && content[1] === 0x5a) { // Windows: PE x86_64/aarch64
  const pePos = content.indexOf('PE\0\0');
  platform = 'win32';
  const code = content[pePos + 4] + (content[pePos + 5] << 8);
  if (code === 0x8664) {
    arch = 'x64';
  } else if (code === 0xaa64) {
    arch = 'arm64';
  } else {
    console.error('Unknown PE arch code:', code.toString(16));
  }
} else if (content[0] === 0xfe && content[1] === 0xed && content[2] === 0xfa && content[3] === 0xce) { // MacOS: Mach-O x86_64/aarch64
  platform = 'darwin';
  if (content[4] === 0x07) {
    arch = 'x64';
  } else if (content[4] === 0x0c) {
    arch = 'arm64';
  }
}

// Create the directory for the platform and architecture if it doesn't exist
if (!fs.existsSync(`bin/${platform}/${arch}`)) {
  fs.mkdirSync(`bin/${platform}/${arch}`, { recursive: true });
}

// Rename the index.node file to executorch.node
fs.renameSync('index.node', `bin/${platform}/${arch}/executorch.node`);

// Copy LLVM libraries for MSVCRT
if (platform === 'win32') {
  const libs = ['libc++.dll', 'libomp.dll', 'libunwind.dll', 'libwinpthread-1.dll'];
  // find install prefix in PATH
  const mingw_prefix = arch === 'arm64' ? 'aarch64-w64-mingw32' : 'x86_64-w64-mingw32';
  const cmdPath = process.env.PATH.split(':').find(p => fs.existsSync(`${p}/${mingw_prefix}-gcc`));
  if (!cmdPath) {
    console.warn('Could not find mingw in PATH');
  }
  const platformDir = path.join(cmdPath, '..', mingw_prefix);
  if (fs.existsSync(platformDir)) {
    for (const lib of libs) {
      fs.copyFileSync(`${platformDir}/bin/${lib}`, `bin/${platform}/${arch}/${lib}`);
    }
  }
}

// Copy the shared libraries to the bin directory
const installPrefix = process.env.EXECUTORCH_INSTALL_PREFIX || 'executorch/cmake-out';

const shared_libs = ['libextension_module', 'qnn_executorch_backend'];

for (const lib of shared_libs) {
  if (fs.existsSync(`${installPrefix}/lib/${lib}.so`)) {
    fs.copyFileSync(`${installPrefix}/lib/${lib}.so`, `bin/${platform}/${arch}/${lib}.so`);
  } else if (fs.existsSync(`${installPrefix}/lib/${lib}.dylib`)) {
    fs.copyFileSync(`${installPrefix}/lib/${lib}.dylib`, `bin/${platform}/${arch}/${lib}.dylib`);
  } else if (fs.existsSync(`${installPrefix}/lib/${lib}.dll`)) {
    fs.copyFileSync(`${installPrefix}/lib/${lib}.dll`, `bin/${platform}/${arch}/${lib}.dll`);
  }
}
