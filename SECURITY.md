# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in Boutquin.Numerics, report it responsibly.

Do not open a public issue for security vulnerabilities.

Please use GitHub Security Advisories to report vulnerabilities privately:
https://github.com/boutquin/Boutquin.Numerics/security/advisories/new

### What to include

- Description of the vulnerability
- Steps to reproduce
- Potential impact (for example: denial-of-service via pathological input, incorrect numerical output with security-relevant consequences, RNG state leakage, deterministic-seed predictability)
- Suggested fix (if any)

### Response timeline

- Acknowledgment: within 48 hours
- Initial assessment: within 7 days
- Fix or mitigation: target within 30 days, depending on severity

### Scope

This policy covers Boutquin.Numerics NuGet packages and source code. Third-party dependencies (for example, the .NET runtime and NuGet dependencies) should be reported to their maintainers.

Numerical inaccuracy that falls within the documented tolerance of an algorithm is not a security vulnerability. Please file such issues through the regular [Issues](https://github.com/boutquin/Boutquin.Numerics/issues) page with a reference value and tolerance context.
