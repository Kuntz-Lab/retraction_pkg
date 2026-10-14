# retraction_pkg

## dVRK dependency

This package talks to the dVRK through `dvrk_python` and `crtk_python_client`, which are built in `~/dvrk_ws` (they are not on PyPI or apt).

Source `~/dvrk_ws/install/setup.bash` before you build and run your workspace. You can also build your package inside `dvrk_ws`.

```bash
source /opt/ros/jazzy/setup.bash
source ~/dvrk_ws/install/setup.bash
cd ~/retraction_ws && colcon build
source install/setup.bash
```
