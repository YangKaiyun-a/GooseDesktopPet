# GooseDesktopPet

GooseDesktopPet is a Windows desktop pet prototype built with WPF. It shows an animated goose on the desktop, supports dragging with the left mouse button, cycles through animation states on click, and exposes a right-click menu with an exit command.

The repository also includes the asset import pipeline used to turn video/WebM character animations into normalized frame assets. The import and segmentation interfaces are intentionally kept open so future builds can support user-uploaded characters, role replacement, and external matting or multimodal AI services.

## Project Layout

- `GooseDesktopPet/` - WPF desktop pet application.
- `GooseDesktopPet/Assets/Pet/` - Built-in normalized goose animation frames and manifest.
- `scripts/import_transparent_webm.py` - Imports already-matted transparent WebM files, normalizes scale/position, flips selected states, trims leading frames, and stabilizes floor-marker x position.
- `scripts/extract_pet_states.py` - Prototype pluggable extraction script for MP4/video sources with multiple segmentation backend hooks, including optional RobustVideoMatting.
- `requirements-tools.txt` - Python dependencies for the asset tools.

## Run

```powershell
dotnet run --project .\GooseDesktopPet\GooseDesktopPet.csproj
```

## Build

```powershell
dotnet build .\GooseDesktopPet\GooseDesktopPet.csproj
```

## Current Interaction

- Left-button drag: move the pet.
- Left-button click: cycle `idle -> action_1 -> action_2 -> idle`.
- `action_2`: performs a left-facing jump with desktop x/y movement.
- Right-button click: opens a context menu. `Exit` is implemented; character replacement is reserved.

## Import A Transparent WebM Pet

Place one transparent WebM per state in a folder. File names become state names, for example:

```text
idle.webm
action_1.webm
action_2.webm
```

Then run:

```powershell
F:\Python\Python311\python.exe .\scripts\import_transparent_webm.py `
  --input-dir . `
  --output-dir .\out_pet `
  --deps-dir .\.python-deps `
  --fps 12 `
  --align-mode frame `
  --scale-mode state-height `
  --reference-state idle `
  --flip-states action_2 `
  --preserve-y-states action_2 `
  --stabilize-marker-x-states action_1 `
  --trim-leading-frames action_2=6 `
  --drop-last-frame
```

Copy the generated `out_pet` contents into `GooseDesktopPet/Assets/Pet`.

## Extension Points

The app includes placeholder contracts in `PetAssetPipeline.cs`:

- `IPetSegmentationService` for background removal / matting.
- `IPetAssetImporter` for turning uploaded videos into app-ready pet manifests.
- `PetSegmentationBackend` for built-in, external, or custom model backends.

The current prototype reads a prepared `manifest.json` from `Assets/Pet`. Later builds can wire the right-click "Change character" menu into these contracts.

## RobustVideoMatting Plugin

RobustVideoMatting is treated as an external matting plugin. It is not bundled into the WPF desktop app and should live in its own Python/model environment.

Expected local layout:

```text
F:\workspace_github\GooseDesktopPet
F:\workspace_github\RobustVideoMatting
```

Download an RVM checkpoint, for example `rvm_mobilenetv3.pth`, then run:

```powershell
F:\Python\Python311\python.exe .\scripts\extract_pet_states.py `
  --backend rvm `
  --input-dir .\source_videos `
  --output-dir .\rvm_extracted_assets `
  --rvm-repo ..\RobustVideoMatting `
  --rvm-checkpoint .\models\rvm\rvm_mobilenetv3.pth `
  --rvm-variant mobilenetv3 `
  --rvm-device cpu `
  --fps 12 `
  --max-frames 48
```

The RVM backend processes each source video as a sequence, writes temporary RGBA frames, then feeds those frames into the same normalization pipeline as the other backends.
