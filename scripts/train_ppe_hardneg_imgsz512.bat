@echo off
setlocal
REM Run from any directory; writes all outputs to local runs/ (do not commit .pt files).
set "ROOT=D:\safety-ai"
set "DATA=%ROOT%\data\ppe_v1_round2\data.yaml"
set "START_WEIGHTS=%ROOT%\runs\ppe_v1_round2_train\finetune_v1\weights\best.pt"

if not exist "%DATA%" (
  echo ERROR: Dataset YAML missing: %DATA%
  exit /b 1
)
if not exist "%START_WEIGHTS%" (
  echo ERROR: Starting checkpoint missing: %START_WEIGHTS%
  echo Edit START_WEIGHTS to the intended checkpoint; do not use a random .pt file.
  exit /b 1
)

cd /d "%ROOT%"
echo Training PPE with imgsz=512; batch=4; workers=0.
echo Confirm hard-negative backgrounds were reviewed and added to TRAIN only.
echo Do not change the resolution to 640 for this run.

yolo detect train model="%START_WEIGHTS%" data="%DATA%" imgsz=512 epochs=20 batch=4 workers=0 device=0 patience=5 project="%ROOT%\runs\ppe_hardneg" name="imgsz512_hardneg_v1" exist_ok=False
if errorlevel 1 (
  echo Training failed. Read the error above; do not assume a new checkpoint exists.
  exit /b 1
)
echo Training command completed. Inspect metrics and false positives before selecting best.pt.
endlocal
