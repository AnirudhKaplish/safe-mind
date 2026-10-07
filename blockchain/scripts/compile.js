// compile.js
// Compiles ConsentManager.sol using the `solc` npm package directly
// (its compiler binary ships INSIDE the npm package as WASM, so no
// separate download from binaries.soliditylang.org is needed - this
// sandbox's network only allows npm registry traffic, not that host).
//
// Output: build/ConsentManager.json containing { abi, bytecode }

const fs = require("fs");
const path = require("path");
const solc = require("solc");

const contractPath = path.join(__dirname, "..", "contracts", "ConsentManager.sol");
const source = fs.readFileSync(contractPath, "utf8");

const input = {
  language: "Solidity",
  sources: {
    "ConsentManager.sol": { content: source },
  },
  settings: {
    outputSelection: {
      "*": { "*": ["abi", "evm.bytecode.object"] },
    },
  },
};

const output = JSON.parse(solc.compile(JSON.stringify(input)));

if (output.errors) {
  const fatal = output.errors.filter((e) => e.severity === "error");
  output.errors.forEach((e) => console.log(e.formattedMessage));
  if (fatal.length > 0) {
    process.exit(1);
  }
}

const contract = output.contracts["ConsentManager.sol"]["ConsentManager"];
const artifact = {
  abi: contract.abi,
  bytecode: "0x" + contract.evm.bytecode.object,
};

const buildDir = path.join(__dirname, "..", "build");
fs.mkdirSync(buildDir, { recursive: true });
fs.writeFileSync(
  path.join(buildDir, "ConsentManager.json"),
  JSON.stringify(artifact, null, 2)
);

console.log("Compiled successfully -> build/ConsentManager.json");
console.log(`ABI entries: ${artifact.abi.length}, bytecode length: ${artifact.bytecode.length}`);
