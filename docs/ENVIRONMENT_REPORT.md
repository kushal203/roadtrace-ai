# RoadTrace AI — Environment Report

## Hardware & Operating System
- **Operating System:** Windows 11 (64-bit)
- **CPU / System RAM:** ~15.2 GB Total Physical Memory
- **Disk Space:** Drive C: ~371.1 GB Free (103.5 GB Used)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU
  - **Total VRAM:** 6141 MiB (~6.0 GB)
  - **Free VRAM:** ~5921 MiB
  - **Driver Version:** 582.05

## Software Stack
- **Python:** Python 3.12.10 (Installed via winget at `C:\Users\yella\AppData\Local\Programs\Python\Python312\python.exe`)
- **Virtual Environment:** `C:\Users\yella\.gemini\antigravity\scratch\roadtrace-ai\venv`
- **PyTorch:** PyTorch 2.5.1 + CUDA 12.1 (`torch-2.5.1+cu121-cp312-cp312-win_amd64.whl`)
- **Torchvision:** torchvision 0.20.1 + CUDA 12.1
- **Node.js:** v24.20.0 (LTS)
- **NPM:** 11.19.0
- **Execution Policy:** Set to `RemoteSigned` for CurrentUser to enable script execution.

## Deployment Target
- **Frontend Framework:** React 18 / 19 + TypeScript + Vite
- **Hosting Platform:** Vercel (Edge/Serverless Static CDN)
- **Browser Inference Engine:** ONNX Runtime Web (`onnxruntime-web`) via WebAssembly (WASM) / WebGL fallback
