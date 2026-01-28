# PyTorch + cuRobo GPU Dev Container

A ready-to-use development container for PyTorch with cuRobo for GPU-accelerated robotics motion generation.

## What's Included

- **PyTorch 2.5.1** with CUDA 12.4 and cuDNN 9
- **cuRobo**: NVIDIA's CUDA-accelerated robot motion generation library
- **Python extensions**: Python, Pylance, Ruff (linting/formatting)
- **Jupyter support**: Full notebook integration in VS Code
- **Common packages**: numpy, pandas, matplotlib, scikit-learn, tensorboard
- **GPU acceleration**: Configured for your NVIDIA GPU
- **Git LFS**: Pre-installed for large file handling

## Quick Start

1. Create your project structure:
   ```
   your-robotics-project/
   ├── .devcontainer/
   │   ├── devcontainer.json
   │   └── setup.sh
   └── (your code here)
   ```

2. Copy `devcontainer.json` and `setup.sh` into the `.devcontainer/` folder

3. Make sure `setup.sh` is executable (it will be handled automatically)

4. Open the folder in VS Code

5. Press `F1` → "Dev Containers: Reopen in Container"

6. **First build takes ~30-40 minutes** (downloads PyTorch image + builds cuRobo)
   - You'll see the progress in the VS Code terminal
   - Subsequent starts are much faster (~10 seconds)

## What Happens During Setup

The container will automatically:
1. Install git-lfs
2. Clone cuRobo from GitHub
3. Install cuRobo with all dependencies (~20 min)
4. Run tests to verify installation
5. Install common Python packages

cuRobo will be available at `/workspace/curobo/` inside the container.

## Verify Installation

Once inside the container, test GPU access and cuRobo:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Test cuRobo import
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
print("cuRobo imported successfully!")
```

## Run cuRobo Examples

cuRobo examples are in `/workspace/curobo/examples/`:

```bash
cd /workspace/curobo/examples
python motion_gen_reacher.py
```

## System Requirements Met

✅ Ubuntu 22.04 (via PyTorch base image)  
✅ NVIDIA GPU with Volta+ architecture (your RTX 5080)  
✅ Python 3.10 (recommended version)  
✅ PyTorch 2.5.1 (>= 1.15 required)  
✅ Git LFS installed

## Configuration Details

- **Shared memory**: 8GB (important for PyTorch DataLoaders)
- **TensorBoard port**: 6006 (forwarded automatically)
- **Format on save**: Enabled with Ruff
- **Python interpreter**: `/opt/conda/bin/python`
- **cuRobo location**: `/workspace/curobo/`

## Customization

### Add more packages
Edit `setup.sh` to add packages after the `pip install` line:
```bash
pip install your-package-here
```

### Use different PyTorch version
Edit the `image` field in `devcontainer.json`:
```json
"image": "pytorch/pytorch:2.4.0-cuda12.1-cudnn8-devel"
```

**Note**: cuRobo requires PyTorch >= 1.15, but 2.0+ is recommended.

## Project Structure Inside Container

```
/workspace/
├── curobo/                    # cuRobo repository
│   ├── examples/              # Example scripts
│   ├── src/curobo/            # cuRobo source code
│   └── tests/                 # Unit tests
└── (your project files)       # Your code goes here
```

## Tips

- **First build is slow** (~30-40 min) but only happens once
- **Rebuild container** if you modify `devcontainer.json`: `F1` → "Dev Containers: Rebuild Container"
- **Access cuRobo examples** at `/workspace/curobo/examples/`
- **Run tests** anytime with: `cd /workspace/curobo && python3 -m pytest .`
- Use Jupyter notebooks directly in VS Code (`.ipynb` files)

## Troubleshooting

**GPU not detected?**
- Verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
- Check `runArgs` includes `"--gpus", "all"`

**cuRobo installation fails?**
- Check the terminal output during container build
- Try rebuilding: `F1` → "Dev Containers: Rebuild Container"
- Manual install: `cd /workspace/curobo && pip install -e . --no-build-isolation`

**Setup script not running?**
- Check that `setup.sh` is in `.devcontainer/` folder alongside `devcontainer.json`
- Rebuild container to re-run setup

**Out of memory errors?**
- Reduce batch size in your code
- Increase `--shm-size` in `runArgs` (currently 8GB)

## Differences from Simple PyTorch Container

- ➕ cuRobo installed and ready to use
- ➕ Git LFS support
- ➕ Setup script for automated installation
- ⏱️ Longer initial build time (~30-40 min vs ~5-10 min)
- 📦 Larger disk usage (~15GB vs ~8GB)

## Next Steps

1. Check out cuRobo documentation: https://curobo.org/
2. Run example scripts in `/workspace/curobo/examples/`
3. Start building your robotics motion planning application!