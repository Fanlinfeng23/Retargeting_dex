# Manus SDK Libraries

This directory is intentionally left without the proprietary shared libraries:

- `libManusSDK.so`
- `libManusSDK_Integrated.so`

They were omitted from GitHub upload preparation for two reasons:

1. each file exceeds GitHub's normal single-file size limit,
2. they are third-party SDK binaries and should be restored locally from the Manus SDK package you already use in the original workspace.

## Official Source

Official MANUS documentation:

- SDK getting started:
  `https://docs.manus-meta.com/3.1.0/Plugins/SDK/getting%20started/`
- Linux guide:
  `https://docs.manus-meta.com/3.1.0/Plugins/SDK/Linux/`
- ROS2 getting started:
  `https://docs.manus-meta.com/3.1.0/Plugins/SDK/ROS2/getting%20started/`
- Downloads index:
  `https://docs.manus-meta.com/latest/Resources/`

The MANUS documentation points users to the MANUS Download Center. After creating a free account or logging in, download:

```text
MANUS Core 3 SDK (including ROS2 Package)
```

The MANUS SDK documentation page shows the package containing Linux examples, Windows examples, and a `ROS2` folder.

## Restore Instructions

After extracting the official SDK package, copy the Linux shared libraries into this directory:

```bash
cp /path/to/MANUS_SDK/ManusSDK/lib/libManusSDK.so \
   /path/to/this/repo/src/ROS2/ManusSDK/lib/

cp /path/to/MANUS_SDK/ManusSDK/lib/libManusSDK_Integrated.so \
   /path/to/this/repo/src/ROS2/ManusSDK/lib/
```

If you rebuild or run the Manus ROS2 publisher on another machine, make sure these libraries are available again before launching the node.

## Notes

- `libManusSDK.so` is the normal SDK library name.
- `libManusSDK_Integrated.so` is used by the integrated Linux workflow documented by MANUS.
- The MANUS Linux guide notes that if you use integrated mode only, you can link directly against `libManusSDK_Integrated.so`, or rename it to `libManusSDK.so` and replace the original in your package directory.
