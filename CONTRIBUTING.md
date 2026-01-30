## Contributing to AI Platform

### Styleguides

#### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to ..." not "Moves cursor to ...")
- Limit the first line to 72 characters or less

#### Code Style Guidelines

- Follow PEP8 conventions (enforced by Ruff)
- Prefer type hints for all public methods and models
- Keep model logic isolated in neurogc/models/
- Avoid side effects in server entry points (server_with_neurogc.py, metrics_server.py)
- Centralize configuration changes in config.json

#### Automatic Changelog Generation

Use [git-cliff](https://git-cliff.org/docs/) for automatic changelog generation

> [!NOTE]
> Always put the changelog in the PR descriptions

```bash
    # git merge-base : to get the base commit
    git merge-base HEAD main | xargs -I{} git-cliff {}..HEAD


    # If you don't want to download git-cliff (as it is from cargo and NOT uv),
    # please use the native git log with formatters like this :
    git merge-base HEAD main | xargs -I{} git log --pretty="- %s" {}..HEAD
```


### Pull Requests

Please generate the changelog and replace that and other details in the [PR Template](./PULL_REQUEST_TEMPLATE.md) given at the root of the project repo.

Kindly lint and format code before raising a PR :

```bash
# Check for issues
ruff check --config pyproject.toml

# Auto-fix
ruff check --fix --config pyproject.toml

# Format
ruff format --config pyproject.toml
```
