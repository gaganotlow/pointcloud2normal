# Project Working Conventions

## Python Environment

Run every Python command for this project in the `point2normal` Conda environment.

Use the non-interactive form in automation and documentation:

```bash
conda run --no-capture-output -n point2normal python <script.py> [args]
```

Do not use the system Python or another Conda environment for training, inference,
web visualization, or MSECNet CUDA-extension builds.
