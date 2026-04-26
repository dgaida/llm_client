# Changelog Workflow

We use [Conventional Commits](https://www.conventionalcommits.org/) to automatically generate our changelog.

## Commit Message Format

Each commit should follow this format:

\`\`\`
<type>(<scope>): <description>
\`\`\`

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Formatting, missing semi-colons, etc.
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Code change that improves performance
- **test**: Adding missing tests
- **chore**: Changes to the build process or auxiliary tools

## Automation

With each release, `git-cliff` is run to update `CHANGELOG.md` based on these commits.
