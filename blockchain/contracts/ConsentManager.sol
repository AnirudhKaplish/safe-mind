// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title ConsentManager
 * @notice Implements the "Dynamic Consent & Blockchain Specifications"
 *         from the approved design:
 *           - Consent Operations: Grant / Modify / Revoke
 *           - Blockchain Data: consent status, purpose/policy metadata,
 *             timestamps/hashes  (NEVER the raw sensitive text itself -
 *             that stays off-chain; only a keccak256 hash/pointer to it
 *             is recorded here for integrity verification)
 *           - Consent Verification: checked BEFORE data processing
 *           - Revocation: blocks any subsequent unauthorized processing
 *
 * Every state-changing action emits an event, which - combined with
 * the immutability of confirmed blockchain transactions - gives us
 * a tamper-evident, timestamped audit trail of every consent decision
 * a user has ever made. A conventional database column could be
 * quietly edited by an admin with DB access; a confirmed transaction
 * on this contract cannot be silently altered after the fact.
 */
contract ConsentManager {
    enum ConsentStatus {
        NeverGranted, // default state - no consent record exists yet
        Granted,
        Revoked
    }

    struct ConsentRecord {
        ConsentStatus status;
        bytes32 purposeHash;     // hash of the purpose/policy text (e.g. "depression-risk-screening-v1")
        bytes32 dataRefHash;     // hash / pointer to the OFF-CHAIN encrypted text - never the text itself
        uint256 lastUpdated;     // block timestamp of the last consent action
    }

    // user address => their current consent record
    mapping(address => ConsentRecord) private consents;

    event ConsentGranted(address indexed user, bytes32 purposeHash, bytes32 dataRefHash, uint256 timestamp);
    event ConsentModified(address indexed user, bytes32 newPurposeHash, bytes32 newDataRefHash, uint256 timestamp);
    event ConsentRevoked(address indexed user, uint256 timestamp);
    event ProcessingAttempt(address indexed user, address indexed processor, bool allowed, uint256 timestamp);

    modifier onlyUser(address user) {
        require(msg.sender == user, "ConsentManager: caller is not the data subject");
        _;
    }

    /// @notice Grant consent for a specific purpose. Can only be called
    ///         by the user themselves (self-sovereign consent).
    function grantConsent(bytes32 purposeHash, bytes32 dataRefHash) external {
        consents[msg.sender] = ConsentRecord({
            status: ConsentStatus.Granted,
            purposeHash: purposeHash,
            dataRefHash: dataRefHash,
            lastUpdated: block.timestamp
        });
        emit ConsentGranted(msg.sender, purposeHash, dataRefHash, block.timestamp);
    }

    /// @notice Modify an existing consent (e.g. change the permitted
    ///         purpose, or point to an updated off-chain data reference).
    ///         Requires that consent was previously granted and not revoked.
    function modifyConsent(bytes32 newPurposeHash, bytes32 newDataRefHash) external {
        require(consents[msg.sender].status == ConsentStatus.Granted, "ConsentManager: no active consent to modify");
        consents[msg.sender].purposeHash = newPurposeHash;
        consents[msg.sender].dataRefHash = newDataRefHash;
        consents[msg.sender].lastUpdated = block.timestamp;
        emit ConsentModified(msg.sender, newPurposeHash, newDataRefHash, block.timestamp);
    }

    /// @notice Revoke consent immediately. From this point on,
    ///         hasValidConsent() returns false and any further
    ///         processing attempt must be blocked by the caller.
    function revokeConsent() external {
        consents[msg.sender].status = ConsentStatus.Revoked;
        consents[msg.sender].lastUpdated = block.timestamp;
        emit ConsentRevoked(msg.sender, block.timestamp);
    }

    /// @notice View function: is this user's consent currently valid?
    function hasValidConsent(address user) public view returns (bool) {
        return consents[user].status == ConsentStatus.Granted;
    }

    /// @notice Returns the full consent record for a user (status,
    ///         purpose hash, data reference hash, last updated time).
    function getConsentRecord(address user) external view returns (ConsentStatus, bytes32, bytes32, uint256) {
        ConsentRecord memory r = consents[user];
        return (r.status, r.purposeHash, r.dataRefHash, r.lastUpdated);
    }

    /// @notice Called by the NLP/processing service BEFORE it processes
    ///         a user's text. Reverts if consent is not currently valid,
    ///         which is how "processing allowed only when valid consent
    ///         exists" is enforced on-chain rather than trusted to the
    ///         application layer alone. Also logs the attempt for audit.
    function checkAndLogProcessing(address user) external returns (bool allowed) {
        allowed = hasValidConsent(user);
        emit ProcessingAttempt(user, msg.sender, allowed, block.timestamp);
        require(allowed, "ConsentManager: processing blocked - no valid consent");
        return allowed;
    }
}
