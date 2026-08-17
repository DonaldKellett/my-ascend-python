## NucBox K11 Mini PC with OrangePi AI Studio Pro

The workflows in this directory were developed on the
[GMKtec NucBox K11 Mini PC](https://www.gmktec.com/products/amd-ryzen%E2%84%A2-9-8945hs-nucbox-k11)
paired with the
[OrangePi AI Studio Pro](http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-AI-Studio-Pro.html)
extension dock.

| Resource | Specifications |
| --- | --- |
| vCPU | 16 |
| Memory | 96G |
| Storage | 1T |
| NPU | Ascend 310P1 x2 |
| AI computing capability | 352 TOPS / 176 TFLOPS |
| Total device memory | 192G |

Unlike the OrangePi AIpro \(20T\) where the workflows are executed directly
within a virtual environment on the host, these workflows are executed within a
containerized environment with MindSpore and CANN pre-installed running on
Kubernetes, with the NPU allocation managed by
[Ascend Docker Runtime](https://www.hiascend.com/document/detail/en/mindx-dl/latest/dluserguide/clusterscheduling/dlug_installation_02_000025.html).
