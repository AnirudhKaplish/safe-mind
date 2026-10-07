// deploy_and_demo.js
//
// Spins up a real local Ethereum-compatible chain (ganache, in-process),
// deploys the compiled ConsentManager contract to it, and walks through
// the exact consent lifecycle from the approved design:
//   1. User grants consent for "mental-health-risk-screening-v1"
//   2. System checks consent BEFORE processing -> allowed
//   3. User modifies consent (e.g. updated purpose / data reference)
//   4. System checks again -> still allowed
//   5. User revokes consent
//   6. System attempts to process again -> BLOCKED (transaction reverts)
//
// Every step prints the real transaction hash and block number, so this
// is genuine on-chain execution, not a simulated/mocked log.

const ganache = require("ganache");
const { ethers } = require("ethers");
const fs = require("fs");
const path = require("path");

const artifact = JSON.parse(
  fs.readFileSync(path.join(__dirname, "..", "build", "ConsentManager.json"), "utf8")
);

function hashString(s) {
  return ethers.keccak256(ethers.toUtf8Bytes(s));
}

async function main() {
  console.log("=".repeat(70));
  console.log("Starting local Ethereum-compatible chain (ganache, in-process)...");
  const server = ganache.provider({ logging: { quiet: true }, chain: { chainId: 1337 } });
  const provider = new ethers.BrowserProvider(server);

  const userSigner = await provider.getSigner(0);   // the data subject (patient/user)
  const processorSigner = await provider.getSigner(1); // the NLP processing service

  const userAddr = await userSigner.getAddress();
  const processorAddr = await processorSigner.getAddress();
  console.log("User (data subject) address     :", userAddr);
  console.log("Processor (NLP service) address :", processorAddr);

  console.log("\nDeploying ConsentManager contract...");
  const factory = new ethers.ContractFactory(artifact.abi, artifact.bytecode, userSigner);
  const contract = await factory.deploy();
  await contract.waitForDeployment();
  const contractAddr = await contract.getAddress();
  console.log("Deployed at:", contractAddr);

  const log = [];

  function record(step, tx, receipt, extra = {}) {
    const entry = {
      step,
      txHash: tx ? tx.hash : null,
      blockNumber: receipt ? receipt.blockNumber : null,
      status: receipt ? (receipt.status === 1 ? "success" : "reverted") : null,
      ...extra,
    };
    log.push(entry);
    console.log(`\n[${step}]`);
    console.log("  tx hash     :", entry.txHash);
    console.log("  block       :", entry.blockNumber);
    console.log("  status      :", entry.status);
    if (Object.keys(extra).length) console.log("  details     :", JSON.stringify(extra));
  }

  // --- Step 1: Grant consent ---
  console.log("\n" + "=".repeat(70));
  console.log("STEP 1: User grants consent");
  const purposeHash1 = hashString("mental-health-risk-screening-v1");
  const dataRefHash1 = hashString("offchain-ref://encrypted-store/user123/post-batch-1");
  const tx1 = await contract.connect(userSigner).grantConsent(purposeHash1, dataRefHash1);
  const rc1 = await tx1.wait();
  record("Grant Consent", tx1, rc1, { purposeHash: purposeHash1 });

  const valid1 = await contract.hasValidConsent(userAddr);
  console.log("  hasValidConsent(user) =", valid1);

  // --- Step 2: Processor checks consent BEFORE processing (allowed) ---
  console.log("\n" + "=".repeat(70));
  console.log("STEP 2: NLP service checks consent before processing (should be ALLOWED)");
  const tx2 = await contract.connect(processorSigner).checkAndLogProcessing(userAddr);
  const rc2 = await tx2.wait();
  record("Processing Check (post-grant)", tx2, rc2, { allowed: true });

  // --- Step 3: User modifies consent ---
  console.log("\n" + "=".repeat(70));
  console.log("STEP 3: User modifies consent (updated purpose/data reference)");
  const purposeHash2 = hashString("mental-health-risk-screening-v2-with-report-sharing");
  const dataRefHash2 = hashString("offchain-ref://encrypted-store/user123/post-batch-2");
  const tx3 = await contract.connect(userSigner).modifyConsent(purposeHash2, dataRefHash2);
  const rc3 = await tx3.wait();
  record("Modify Consent", tx3, rc3, { newPurposeHash: purposeHash2 });

  // --- Step 4: Processor checks again (still allowed) ---
  console.log("\n" + "=".repeat(70));
  console.log("STEP 4: NLP service checks consent again (should still be ALLOWED)");
  const tx4 = await contract.connect(processorSigner).checkAndLogProcessing(userAddr);
  const rc4 = await tx4.wait();
  record("Processing Check (post-modify)", tx4, rc4, { allowed: true });

  // Snapshot the "still granted" state here for report-generator demo
  // purposes (before we revoke in step 5) - lets us show BOTH the
  // allowed and blocked report-generation paths.
  const grantedSnapshotPath = path.join(__dirname, "..", "..", "results", "blockchain_consent_state_granted.json");
  fs.writeFileSync(grantedSnapshotPath, JSON.stringify({ contractAddress: contractAddr, steps: log, blockedConfirmed: false }, null, 2));
  console.log(`  (snapshot of still-granted state saved to ${grantedSnapshotPath})`);

  // --- Step 5: User revokes consent ---
  console.log("\n" + "=".repeat(70));
  console.log("STEP 5: User REVOKES consent");
  const tx5 = await contract.connect(userSigner).revokeConsent();
  const rc5 = await tx5.wait();
  record("Revoke Consent", tx5, rc5);

  const valid2 = await contract.hasValidConsent(userAddr);
  console.log("  hasValidConsent(user) =", valid2, " (expected: false)");

  // --- Step 6: Processor attempts to process again -> should be BLOCKED ---
  console.log("\n" + "=".repeat(70));
  console.log("STEP 6: NLP service attempts to process again after revocation");
  console.log("        (this transaction is EXPECTED to revert)");
  let blockedConfirmed = false;
  try {
    const tx6 = await contract.connect(processorSigner).checkAndLogProcessing(userAddr);
    await tx6.wait();
    console.log("  UNEXPECTED: transaction succeeded (this should not happen)");
  } catch (err) {
    blockedConfirmed = true;
    const reason = err.shortMessage || err.message;
    console.log("  Transaction reverted as expected.");
    console.log("  Revert reason:", reason);
    log.push({ step: "Processing Check (post-revoke)", txHash: null, status: "reverted-as-expected", reason });
  }

  // --- Final summary ---
  console.log("\n" + "=".repeat(70));
  console.log("SUMMARY");
  console.log(`  Consent granted, verified, modified, verified again, then revoked.`);
  console.log(`  Post-revocation processing attempt was blocked: ${blockedConfirmed}`);

  const outPath = path.join(__dirname, "..", "..", "results", "blockchain_consent_demo_log.json");
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify({ contractAddress: contractAddr, steps: log, blockedConfirmed }, null, 2));
  console.log(`\nFull demo log saved to: ${outPath}`);

  if (typeof server.disconnect === "function") {
    await server.disconnect();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
