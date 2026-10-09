# Finger Studio

A Windows desktop app that uses a webcam and MediaPipe Hands to trace selected fingertips.

## Setup

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   python main.py
   ```

The MediaPipe hand model is included in `tracer/hand_landmarker.task`, so normal startup does not require a model download.

Alternatively, on Windows double-click `start_tracer.bat`. It creates a local virtual environment, installs dependencies, and starts the application.

## Controls

- **Active hand**: choose Either, Left, or Right. The selection never silently changes.
- **Holographic two-hand selector**: the studio uses the supplied side-by-side open-palm hand artwork with a restrained cyan hologram treatment. The artwork keeps a clear gap between the hands, and each visible finger has an anatomically aligned clickable region, hover state, selected state, and keyboard shortcut (1–5). Selecting a finger immediately updates the real tracker.
- **Spatial layout**: the interface uses airy layered surfaces, soft lavender/navy contrast, restrained borders, depth through elevation, and a calmer control-center hierarchy around the camera workspace rather than brutalist hard-edged panels.
- **Start / Pause / Resume / Stop**: control trail recording.
- **Clear trail**: remove all recorded segments.
- **Thickness / Color**: customize the trail.
- **Show landmarks**: toggle the hand skeleton and fingertip markers.
- **Mirror preview**: keep the natural selfie orientation, or disable it for a non-mirrored view.
- **Camera range**: the complete camera frame (0-1 on every side) is used by default. Adjust the normalized left, top, right, and bottom values only when you want a smaller active region, or restore the full frame with **Use full camera frame**.
- **Camera resolution**: the app requests a 1920×1080 full-frame feed instead of limiting capture to 640×480. The preview and normal landmark inference use the complete camera frame. If no hand is found, one enlarged recovery pass is attempted for small fingers near the frame edge.
- **Resizable layout**: drag the divider between the camera and controls to give either side more space.

The trace is intentionally broken after tracking loss or an implausible fingertip jump, preventing unwanted connecting lines. The application keeps a bounded trail history. MediaPipe runs in video mode so it reuses temporal tracking between frames, uses permissive confidence gates for cropped/partly occluded palms, and drops stale frames rather than building a backlog.

The selector hit regions are defined in the supplied artwork's native 738×414 coordinate system and transformed with the exact uniform scale used to display the image. They follow the visible finger silhouettes rather than using broad rectangular zones, including the separated pinkies and inward-facing thumbs.

## Limitations

Accuracy depends on lighting, camera quality, hand orientation, and occlusion. The bundled MediaPipe Hand Landmarker is a hand-landmark model, not a single-finger detector: if only a fingertip is visible and the palm/wrist context is completely outside the frame, no landmark model can reliably infer the hidden finger identity. The enlarged recovery pass improves small or edge-cropped hands, while severe motion blur and a completely isolated fingertip remain limitations. This version does not control the system mouse; cursor control is intentionally out of scope.

## Tests

Run the core behavior tests with:

```powershell
python -m unittest discover -s tests -v
```
