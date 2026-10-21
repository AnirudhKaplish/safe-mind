// consent_server.js
// A small always-on bridge between the user web app and the REAL ConsentManager contract.
// It starts an in-process ganache chain, deploys the compiled contract once, and exposes
// plain HTTP endpoints on 127.0.0.1:8546. Every call is a real transaction with a real hash.
//   GET  /status   -> { ok, address, hasValidConsent }
//   POST /grant    -> grantConsent(...)
//   POST /modify   -> modifyConsent(...)
//   POST /revoke   -> revokeConsent(GAS)
//   POST /check    -> checkAndLogProcessing(user)   (reverts if consent is not valid)
// Uses only ganache + ethers, which are already installed. No extra packages.
const http = require("http");
const fs = require("fs");
const path = require("path");
const ganache = require("ganache");
const { ethers } = require("ethers");

const PORT = process.env.CONSENT_PORT || 8546;
const artifact = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "build", "ConsentManager.json"), "utf8"));
const h = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));

let contract, userSigner, procSigner, userAddr, contractAddr;
const history = [];
// Explicit gas limit: after one reverted estimateGas, ethers + ganache otherwise keep failing.
const GAS = { gasLimit: 300000 };

async function boot() {
  const server = ganache.provider({ logging: { quiet: true }, chain: { chainId: 1337 } });
  const provider = new ethers.BrowserProvider(server);
  userSigner = await provider.getSigner(0);
  procSigner = await provider.getSigner(1);
  userAddr = await userSigner.getAddress();
  const factory = new ethers.ContractFactory(artifact.abi, artifact.bytecode, userSigner);
  contract = await factory.deploy();
  await contract.waitForDeployment();
  contractAddr = await contract.getAddress();
  console.log("ConsentManager deployed at", contractAddr);
}

async function send(label, fn) {
  try {
    const tx = await fn();
    const rc = await tx.wait();
    const entry = { ok: true, action: label, txHash: tx.hash, block: rc.blockNumber, at: new Date().toISOString() };
    history.push(entry);
    return entry;
  } catch (e) {
    const entry = { ok: false, action: label, reverted: true, reason: e.shortMessage || e.message, at: new Date().toISOString() };
    history.push(entry);
    return entry;
  }
}

async function handle(route) {
  switch (route) {
    case "/status": return { ok: true, address: contractAddr, hasValidConsent: await contract.hasValidConsent(userAddr), history: history.slice(-10) };
    case "/grant": return send("grant", () => contract.connect(userSigner).grantConsent(h("mental-health-risk-screening-v1"), h("offchain-ref://user-app/session-" + Date.now()), GAS));
    case "/modify": return send("modify", () => contract.connect(userSigner).modifyConsent(h("mental-health-risk-screening-v2-with-report-sharing"), h("offchain-ref://user-app/session-" + Date.now()), GAS));
    case "/revoke": return send("revoke", () => contract.connect(userSigner).revokeConsent(GAS));
    case "/check": return send("check", () => contract.connect(procSigner).checkAndLogProcessing(userAddr, GAS));
    default: return { ok: false, error: "unknown route" };
  }
}

boot().then(() => {
  http.createServer(async (req, res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Content-Type", "application/json");
    if (req.method === "OPTIONS") { res.end("{}"); return; }
    try { res.end(JSON.stringify(await handle(req.url.split("?")[0]))); }
    catch (e) { res.statusCode = 500; res.end(JSON.stringify({ ok: false, error: String(e) })); }
  }).listen(PORT, "127.0.0.1", () => console.log("Consent bridge listening on http://127.0.0.1:" + PORT));
}).catch((e) => { console.error("Failed to start chain:", e); process.exit(1); });
