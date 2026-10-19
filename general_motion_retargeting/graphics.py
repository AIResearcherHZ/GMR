import os
import sys

import glfw
import mujoco as mj


def configure_graphics():
    if sys.platform != "linux":
        return
    if os.environ.get("DISPLAY"):
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    if os.path.exists("/proc/driver/nvidia/version"):
        os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
        if os.environ["__GLX_VENDOR_LIBRARY_NAME"] == "nvidia" and mj.get_mju_user_warning() is None:
            mj.set_mju_user_warning(_graphics_warning)


def _graphics_warning(message):
    if message == "OpenGL error 0x502 in or before mjr_makeContext":
        return
    print(f"WARNING: {message}", file=sys.stderr, flush=True)
