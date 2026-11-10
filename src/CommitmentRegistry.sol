// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// Day-ahead commitments of certifiable energy and their settlement on metered data (H4).
/// Amounts are integers in micro p.u.*h. The contract stores; it does not judge the forecast.
contract CommitmentRegistry {
    struct Entry { uint64 committed; uint64 certified; uint64 shortfall; bool isCommitted; bool isSettled; }

    address public admin;
    mapping(address => bool) public producer;
    mapping(address => bool) public adapter;
    mapping(bytes32 => mapping(uint32 => Entry)) public entries; // method => day number => entry

    event Committed(bytes32 indexed method, uint32 indexed day, uint64 amount, bytes32 inputsHash);
    event Settled(bytes32 indexed method, uint32 indexed day, uint64 certified, uint64 shortfall, bytes32 meterHash);

    constructor() { admin = msg.sender; }

    function setProducer(address who, bool on) external { require(msg.sender == admin, "admin"); producer[who] = on; }
    function setAdapter(address who, bool on) external { require(msg.sender == admin, "admin"); adapter[who] = on; }

    function commit(bytes32 method, uint32 day, uint64 amount, bytes32 inputsHash) external {
        require(producer[msg.sender], "producer");
        Entry storage e = entries[method][day];
        require(!e.isCommitted, "already committed");
        require(!e.isSettled, "already settled");
        e.committed = amount;
        e.isCommitted = true;
        emit Committed(method, day, amount, inputsHash);
    }

    function settle(bytes32 method, uint32 day, uint64 certified, bytes32 meterHash) external {
        require(adapter[msg.sender], "adapter");
        Entry storage e = entries[method][day];
        require(e.isCommitted, "no commitment");
        require(!e.isSettled, "already settled");
        e.certified = certified;
        e.shortfall = e.committed > certified ? e.committed - certified : 0;
        e.isSettled = true;
        emit Settled(method, day, certified, e.shortfall, meterHash);
    }
}
