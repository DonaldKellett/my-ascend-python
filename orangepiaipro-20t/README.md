## OrangePi AIpro \(20T\)

The workflows in this directory were developed on [OrangePi AIpro \(20T\)](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AIpro%2820t%29.html) with version 2025-09-09 of the official Orange Pi Ubuntu 22.04 LTS desktop image using the default libraries and package versions unless otherwise specified.

| Image | MindSpore version | CANN version |
| --- | --- | --- |
| `opiaipro_20t_ubuntu22.04_desktop_aarch64_20250909.img.xz` | `2.4.10` | `8.0.0` |

### Packages installed in `base` Python virtual environment

The default packages installed in the `base` Python virtual environment are specified in `requirements.txt`. Workflows in this directory assume the `base` virtual environment unless otherwise specified.

Workflows depending on alternative packages specify an alternative `requirements.txt` within their own subdirectory. It is recommended to create a dedicated virtual environment for these alternative packages instead of modifying the `base` directly.
