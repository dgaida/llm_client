# Bug Report: Multiple Version Lines Cause Tagging Failure

## Title
GitHub Action fails to create git tags when `pyproject.toml` contains multiple "version" entries.

## Summary
The `dgaida/auto-version-action` fails when used in a repository where `pyproject.toml` has more than one line starting with `version = "`. This typically happens when using tools like `commitizen` which maintain their own version configuration section in the same file. The action extracts multiple version strings into a single variable, leading to invalid `git tag` commands.

## Steps to Reproduce
1. Create a `pyproject.toml` with both a `[project]` section and a `[tool.commitizen]` section, both containing `version = "X.Y.Z"`.
2. Run the `auto-version-action`.
3. Observe the output log.

## Observed Behavior
The action logs something like:
```
To https://github.com/dgaida/llm_client
   738ef46..8a2ee90  master -> master
fatal: Failed to resolve '0.4.7' as a valid ref.
Error: Process completed with exit code 128.
```
This happens because the extraction logic:
```bash
NEW_VERSION=$(grep '^version = "' pyproject.toml | cut -d'"' -f2)
```
returns a multi-line string if multiple matches exist:
```
0.4.8
0.4.7
```
When the action later runs:
```bash
git tag v$NEW_VERSION
```
The shell expands this to `git tag v0.4.8 0.4.7`. Git interprets `0.4.7` as the commit/ref to be tagged, but since it doesn't exist, the command fails.

## Technical Analysis
1.  **Brittle Grep**: The regex `^version = "` is too broad and matches any top-level or section-level `version` key that starts a line.
2.  **Missing Line Limit**: The `grep` command does not limit itself to the first match.
3.  **Unquoted Variables**: Shell variables like `$NEW_VERSION` are not quoted in the action's shell script, causing word splitting when they contain whitespace or newlines.
4.  **Desynchronization**: The `increment_version.py` script in the action only updates the *first* occurrence of the version string, leaving other sections (like `[tool.commitizen]`) with outdated version numbers.

## Proposed Fixes

### 1. Robust Version Extraction (action.yml)
Update the extraction logic to take only the first match and use proper quoting for variables:
```bash
NEW_VERSION=$(grep '^version = "' pyproject.toml | head -n 1 | cut -d'"' -f2)
if [ -n "$NEW_VERSION" ]; then
  if ! git rev-parse "v$NEW_VERSION" >/dev/null 2>&1; then
    git tag "v$NEW_VERSION"
    git pus""h origin "v$NEW_VERSION"
  fi
fi
```

### 2. Synchronized Version Increment (increment_version.py)
Modify the Python script in the action to:
1. Find the current version from the first match.
2. Calculate the new incremented version.
3. Replace **all** occurrences of `version = "X.Y.Z"` in the file with the new version string to ensure consistency across all configuration sections.
