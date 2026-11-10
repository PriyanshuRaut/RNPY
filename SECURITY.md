# Security Policy

## Supported Versions

Security updates are provided only for the latest stable release of this project.

| Version | Supported |
| ------- | --------- |
| Latest  | ✅ Yes |
| Older releases | ❌ No |

Users are strongly encouraged to upgrade to the latest version to receive security fixes.

---

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

### How to Report
- Do **not** open a public GitHub issue
- Send details privately via the contact method listed in the Code of Conduct
- Include:
  - A clear description of the vulnerability
  - Steps to reproduce (if possible)
  - Potential impact

### What to Expect
- You can expect an initial response within **7 days**
- If the issue is accepted, a fix will be worked on and released as soon as possible
- If the issue is declined, an explanation will be provided

Thank you for helping keep this project secure.

## Development versions and diagnostic privacy

The v0.4 development checkout is pre-release work, not a supported production
release. NPGuard collects metadata locally and does not upload telemetry or
array contents. Metadata can include local filenames, line numbers, labels,
dtypes, shapes, thread IDs, object IDs, and sizes. Review and redact it before
sharing reports or benchmark files. Avoid sensitive information in labels.
Application exceptions are represented by their type name; their messages and
tracebacks are not retained in observation results. Array values are not read
for diagnosis, and arrays/sources are not kept alive by the registry.

The normal installation/build process uses the Python package index and executes
package build backends. Use trusted dependency sources and keep TLS verification
enabled. Report suspected vulnerabilities privately using the contact method
above; do not include credentials or sensitive application data in public issues.
