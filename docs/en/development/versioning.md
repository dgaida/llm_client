# Versioning

This project uses **Semantic Versioning (SemVer)** and automates the release process using `mike`.

## Mike (Versioned Documentation)

We use `mike` to provide multiple versions of the documentation simultaneously.

### Deploying a New Version

To publish a new version of the documentation:

```bash
# Mike deployment for a new version
mike deploy --push --update-aliases 0.5.0 latest
# Set default version
mike set-default --push latest
```

### Local Preview

Preview a specific version:

```bash
mike serve
```

## Automation

Package versioning is controlled via GitHub Actions (`auto-version.yml`) based on conventional commits.
