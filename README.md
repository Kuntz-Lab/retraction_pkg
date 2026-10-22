# retraction_pkg

## Dependencies

### dVRK

This package talks to the dVRK through `dvrk_python` and `crtk_python_client`, which are built in `~/dvrk_ws` (they are not on PyPI or apt). Source `~/dvrk_ws/install/setup.bash` before running the node.

### PushVIBES

The node imports the push predictor (`deploy_push.py`) from the PushVIBES checkout at `~/retraction_ws/src/PushVIBES`. PushVIBES is a flat repo of scripts rather than a package (its modules import each other as `from bps import ...`), so the node adds the repo root to `sys.path` (`PUSHVIBES_ROOT` at the top of [retraction_experiments_node.py](retraction_pkg/retraction_experiments_node.py)). Pass absolute paths for checkpoints and BPS bases, since `DeployPushVIBES`'s defaults are relative to the PushVIBES directory.

## Python environment

The node runs in **PushVIBES's venv, `~/retraction_ws/src/PushVIBES/.pushvibes_venv`**. There is no separate venv for this package. The same venv is used for PushVIBES training and its gold-standard tests.

The venv must be built with `--system-site-packages`. `rclpy` needs PyYAML and the node needs PyKDL, both of which come from apt (`/usr/lib/python3/dist-packages`) and are hidden from a plain venv. Deactivate conda first, so the venv is built from the system Python:

```bash
conda deactivate   # if a conda env is active
cd ~/retraction_ws/src/PushVIBES
/usr/bin/python3 -m venv --system-site-packages .pushvibes_venv
source .pushvibes_venv/bin/activate
pip install -r requirements.txt
```

PushVIBES needs numpy 2, but the apt numpy is 1.26, and some apt packages with compiled code (matplotlib, contourpy, scipy) were built against it. If the venv loaded those, they would crash against its numpy 2. `requirements.txt` pins version floors that the apt copies can't satisfy, so pip installs venv copies of them instead. Check that matplotlib comes from the venv:

```bash
python -c "import matplotlib, contourpy; print(matplotlib.__file__); print(contourpy.__file__)"
# both paths should be inside .pushvibes_venv
```

After rebuilding or changing the venv, run the PushVIBES gold-standard tests (see `src/PushVIBES/CLAUDE.md`).

## Build

Build in a plain shell. The venv doesn't need to be active, but conda must be deactivated, or CMake uses conda's Python to generate `retraction_interface`:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/retraction_ws && colcon build
```

The launcher's shebang is `#!/usr/bin/env python3` (set in [setup.cfg](setup.cfg)), so it runs on whichever `python3` is first on `PATH` at run time. That is the venv's Python once the venv is activated. Don't use `--symlink-install`. It ignores `setup.cfg` and hard-codes the Python that ran colcon. If you need it, copy the `CommandSpec` patch from the top of `diffdef_pkg/setup.py`.

## Running

In a new terminal, activate the venv first:

```bash
source ~/retraction_ws/src/PushVIBES/.pushvibes_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/dvrk_ws/install/setup.bash
source ~/retraction_ws/install/setup.bash
ros2 run retraction_pkg retraction_experiments_node
```

PSM1 must be enabled and homed in the dVRK console. Otherwise the dVRK ignores `move_cp` and the arm won't move. Check with:

```bash
ros2 topic echo --once /PSM1/operating_state   # state: ENABLED, is_homed: true
```

## Troubleshooting

| Error | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'yaml'` (or `PyKDL`) | The venv was built without `--system-site-packages`. | Rebuild the venv as above. |
| `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` | An apt package with compiled code that uses numpy is shadowing the venv. The last `/usr/lib/python3/dist-packages/<pkg>` line in the traceback names it. | Add a version floor for `<pkg>` to `PushVIBES/requirements.txt` and reinstall. |
| `move_cp, arm not ready` on `/PSM1/error` | PSM1 isn't enabled and homed. | Power on and home PSM1 from the console. |
