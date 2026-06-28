# autolabel
autolabel leverages super resolution


# Auto-Labeling Pre-Processor: Super Resolution Pipeline

Sequential upscaling pipeline using **Real-ESRGAN** (Images) and **RealBasicVSR** (Videos).

## Setup Instructions

1. **Clone the Sub-Repositories**
   ```bash
   mkdir -p repositories
   git clone [https://github.com/](https://github.com/)<your-username>/Real-ESRGAN.git repositories/Real-ESRGAN
   git clone [https://github.com/](https://github.com/)<your-username>/realbasicVSR.git repositories/realbasicVSR

<img width="587" height="317" alt="Screenshot 2026-06-28 at 12 48 18" src="https://github.com/user-attachments/assets/7fbf94af-81d2-4f88-b681-1c3f1e4cb7f3" />


autolabel/
│
├── repositories/          # Clone your two forked repos here
│   ├── Real-ESRGAN/
│   └── realbasicVSR/
│
├── weights/               # Place your downloaded model checkpoints here
│   ├── RealESRGAN_x4plus.pth
│   └── RealBasicVSR_x4.pth
│
├── config.yaml            # Central configuration file
├── processors.py          # Wrappers for running repo inference
├── main.py                # Main pipeline orchestrator
└── README.md              # Documentation
