# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub Security Advisories](https://github.com/SimonCrouzet/EvoFlowNet/security/advisories/new),
not as a public issue.

Expect an acknowledgement within a week. If a report is confirmed, the fix and
the advisory are published together.

## Supported versions

This project is pre-1.0. Only the latest release receives fixes.

## Scope

This is a research library for sequence optimisation. It has no network
services, no authentication and no user data, so the realistic attack surface
is narrower than for most software. The things worth reporting:

- **Deserialisation.** Loading a checkpoint, a cached landscape or a campaign
  state file from an untrusted source in a way that can execute code.
- **Dataset download paths.** Anything that writes outside the intended cache
  directory, or that accepts a downloaded file without verifying its checksum.
- **Supply chain.** A dependency or pinned GitHub Action that is compromised
  or resolves to something unexpected.
- **Configuration loading.** Hydra and OmegaConf resolve expressions in config
  files; a path where a config from an untrusted source gains more capability
  than it should.

## Not vulnerabilities

To be explicit, since this is a scientific library:

- A method producing poor results, failing to converge, or being outperformed
  by a baseline. Those are bugs or research findings -- please open an issue.
- Resource exhaustion from a configuration you supplied yourself. Sampling is
  meant to be run with parameters that can consume a large amount of compute.
