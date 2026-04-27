# Manus SDK Libraries

This directory is intentionally left without the proprietary shared libraries:

- `libManusSDK.so`
- `libManusSDK_Integrated.so`

They were omitted from GitHub upload preparation for two reasons:

1. each file exceeds GitHub's normal single-file size limit,
2. they are third-party SDK binaries and should be restored locally from the Manus SDK package you already use in the original workspace.

## Restore Instructions

Copy the local SDK binaries back into this directory from your existing machine setup:

```bash
cp /home/user/ros2_ws/src/ROS2/ManusSDK/lib/libManusSDK.so \
   /path/to/this/repo/src/ROS2/ManusSDK/lib/

cp /home/user/ros2_ws/src/ROS2/ManusSDK/lib/libManusSDK_Integrated.so \
   /path/to/this/repo/src/ROS2/ManusSDK/lib/
```

If you rebuild or run the Manus ROS2 publisher on another machine, make sure these libraries are available again before launching the node.
