### Setup

1. Clone the repo :

   ```bash
       git clone https://github.com/contentstack/aip
       cd aip
   ```

2. Create virtual environment using [uv](https://docs.astral.sh/uv/getting-started/installation/) :

   ```bash
       uv venv --python 3.14.2
   ```

3. Install dependecies, run following command from root directory.

   ```bash
       uv sync

       # To upgrade all packages to their latest compatible versions
       uv sync --upgrade
   ```

4. Linting

   ```bash
      # show fixes (do this)
      ruff check --config pyproject.toml

      # should be enough (do this).
      ruff check --fix --config pyproject.toml

      ruff check --unsafe-fixes --config pyproject.toml        # only if necessary
      ruff check --fix --unsafe-fixes --config pyproject.toml  # only if necessary
   ```

> [!NOTE]  
> Manually correct your code and abide to whatever `ruff check` suggests.
> Running `ruff check --fix` can only do so much.

5. Formatting

   ```bash
      ruff check --select I --fix . --config pyproject.toml
      ruff format --config pyproject.toml
   ```

> [!NOTE]
> We do the `check` step as well to enforce `isort` (import sort) as well
> while formatting

### Usage

#### Phase 1 : Collect Training Data (without NeuroGC)

```bash
python server_without_neurogc.py

# This will also serve the UI at localhost:{metrics_server["port"]}
python metrics_server.py

# This generates profiler_data.csv with at least 120 rows of data.
TARGET_SERVERS=without_gc locust -f locustfile.py --headless -u 10 -r 2 -t 2m
```

#### Phase 2 : Train the ML model

```bash
# Train using the collected data
# Output : gc_model.pth (trained model weights)
python model.py --train profiler_data.csv
```

#### Phase 3: Run Both Servers for Comparison

```bash
python server_with_neurogc.py
python server_without_neurogc.py
python metrics_server.py
# Load test BOTH servers
locust -f locustfile.py --headless -u 20 -r 5 -t 5m
```

