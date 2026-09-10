# Insta360 ONE X stitcher and viewer

Turns the `.insp` photos an Insta360 ONE X writes into finished 360°
panoramas and lets you look around in them. Point it at a folder, press
*Stitch*, drag to look around. Two front ends share the same method:

- **Web app** (`docs/`): runs entirely in the browser, hosted as static files
  on GitHub Pages. Nothing is uploaded; photos are read from your disk and
  processed with WebGL 2 plus plain JavaScript. No install.
- **Desktop app** (`insta360stitch/`): Python, Tkinter and OpenCV for Linux and
  Windows, with a batch command line. The reference implementation.

The processing follows the Hugin workflow, automated:

| Hugin step | What this app does |
|---|---|
| Load images, set lens type / FOV | Reads the factory lens calibration stored in the `.insp` trailer (centres, radii, rear-lens rotation) and starts from a 200° equidistant fisheye model. |
| Create control points, optimise `v`, `y/p/r` | Finds SIFT matches in the two overlap zones and refines the lens FOV and the rear lens' yaw/pitch/roll with a robust Levenberg–Marquardt fit (typically ~150 points, 0.5° RMS). The fitted model is cached per camera serial. |
| Level the panorama | Uses the accelerometer samples stored in the file to put the horizon straight. |
| Photometric optimisation | Per-channel gain match measured in the overlap. |
| enblend: seam finder + multi-band blending | Minimum-cost seam through the overlap band, Laplacian-pyramid blend across it. |
| *(no Hugin equivalent)* parallax | Dense optical flow (DIS) between the two lenses in each seam band, forward/backward checked, warps both lenses half way toward each other around the seam so nearby objects line up instead of ghosting. |

## Why the parallax pass

The two lenses of a ONE X sit about 2.5 cm apart. Any single projection
(the fixed remap every quick stitcher uses) can align the images at one
distance only; things closer than a few metres show up in different places in
each lens and ghost or tear along the seam. The flow pass measures that local
displacement and warps the two images into agreement in a band around the seam
only, so the rest of the sphere keeps its true geometry. Regions where the flow
is unreliable (occlusions, the hand holding the camera) fall back to the plain
seam.

## Web app (GitHub Pages)

The `docs/` folder is a complete static site. To publish it, enable GitHub
Pages for the repository with *Deploy from a branch*, choose the branch and
the `/docs` folder, and open the URL GitHub shows. To try it locally, serve
the folder with any static server, for example:

```bash
python -m http.server --directory docs 8000     # then open http://localhost:8000
```

How to use it:

1. *Open folder…* picks a folder (Chrome and Edge open it directly and can
   write results back into a `stitched/` sub-folder; Firefox and Safari use
   the browser's folder chooser and the results are kept in memory for
   *Download panorama*). *Open files…* picks individual `.insp` files.
2. Select a photo, press *Stitch selected* or double-click it; *Stitch all*
   works through the folder. Progress shows in the status bar.
3. Drag to look around, wheel or pinch to zoom, arrows to pan, `F` for the
   flat equirectangular view, double-click to reset.

The width menu only offers sizes the GPU can hold; 4096 is the default, 6080
the camera's native size. Stitching takes a second or two on a desktop GPU at
4096. A file's IMU data is borrowed from a bracket sibling (same
`IMG_<date>_<time>` prefix) when the file itself has none, exactly like the
desktop app. The refined lens model is remembered per camera serial in the
browser's local storage.

What runs where: the fisheye-to-equirectangular projection, exposure gains,
compositing and levelling are WebGL 2 fragment shaders; the seam work
(optical flow, seam search, Laplacian blend) is typed-array JavaScript on the
two narrow seam bands only. The optical flow is a coarse-to-fine block
matcher with zero-mean SAD, sub-pixel refinement and a forward/backward
check, and the lens refinement fits FOV and rear-lens rotation to the
forward/backward-consistent flow vectors with the same Cauchy-weighted
Levenberg–Marquardt as the desktop app. The two implementations agree to
within about 2.5 grey levels on the same input at the same size.

## Desktop app: install and run

Requirements: Python 3.9 or newer with Tkinter (included with the python.org
Windows installer; on Linux install `python3-tk` from your distribution).

**Windows**: double-click `run_app.bat`. It creates a virtual environment in
`.venv` on first run and installs the three dependencies (numpy, OpenCV,
Pillow), then opens the app.

**Linux**:

```bash
./run_app.sh                  # first run creates .venv and installs dependencies
./run_app.sh /path/to/photos  # open a folder straight away
```

Or manually on either platform:

```bash
python -m pip install -r requirements.txt
python run_app.py [folder]
```

## Using the app

1. *Open folder…* and choose the directory with your `.insp` files (the
   `DCIM/Camera01` folder of the camera works directly).
2. Select a file. Unstitched files show the raw dual-fisheye picture; stitched
   ones open in the interactive viewer.
3. *Stitch selected* (or double-click) / *Stitch all*. Results go to a
   `stitched/` sub-folder as JPEGs with GPano XMP metadata, so Google Photos,
   Facebook, VLC and similar viewers recognise them as 360° images. A `.json`
   sidecar records the lens model and seam statistics.
4. In the viewer: drag to look around, mouse wheel to zoom, arrow keys to pan,
   `F` toggles the flat equirectangular view, double-click resets. *Save
   view…* writes the current rectilinear view to a file.

Options in the toolbar:

- **Width** – output width (height is always half). 6080 is the camera's native
  resolution; 4096 is a good compromise for speed.
- **Parallax fix** – the optical-flow seam pass. Turn it off to see what a
  fixed projection does.
- **Auto level** – horizon correction from the IMU.
- **Refine lenses** – control-point optimisation of the lens model. With it
  off the factory calibration is used as is.

## Command line

```bash
python -m insta360stitch stitch /path/to/photos            # everything in the folder -> photos/stitched/
python -m insta360stitch stitch IMG_001.insp -w 4096 -o out
python -m insta360stitch stitch photos --no-parallax --no-level --force
python -m insta360stitch info photos                       # show the metadata found in the files
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests render a synthetic panorama through the lens model into a fake
`.insp` file (JPEG plus Insta360 trailer), parse it back and check that the
stitcher reproduces the source, that the refinement recovers a deliberate
rear-lens yaw error, and that levelling and the file parser behave.

## File format notes

An `.insp` is a JPEG (6080×3040, the two fisheye circles side by side)
followed by a trailer of records, each ending in a `uint16 id, uint32 length`
footer, then 32 zero bytes, the trailer length, a version and the magic string
`8db42d694ccc418790edff439fe026bf`. Record `0x0101` is a protobuf with the
serial number, model, firmware and the calibration string
`2_cx1_cy1_r1_yaw1_pitch1_roll1_cx2_cy2_r2_yaw2_pitch2_roll2_w_h_…`
(expressed in the sensor's portrait frame, so the axes are swapped relative to
the stored JPEG). Record `0x0200` is a thumbnail JPEG and `0x0300` holds IMU
samples (`uint64` timestamp, 3 doubles accelerometer, 3 doubles gyro).

## Layout

```
docs/            the web app (GitHub Pages)
  index.html, style.css
  js/insp.js     .insp parser
  js/geometry.js lens model, rotations
  js/gl.js       WebGL 2 helper and the projection / levelling / viewer shaders
  js/imageops.js typed-array image operations (blur, pyramids, resampling)
  js/flow.js     block-matching optical flow + forward/backward check
  js/calibrate.js lens refinement (control points from flow, robust LM)
  js/stitch.js   pipeline: bands, gains, seams, parallax warp, blend, level
  js/jpeg.js     JPEG with Exif + GPano XMP
  js/viewer.js   360 viewer
  js/app.js      UI, folder access, queue, downloads
insta360stitch/
  insp.py        .insp parser (JPEG, trailer records, calibration, IMU)
  geometry.py    rotations, equirectangular grid, fisheye lens model, renderer
  calibrate.py   control points + robust optimisation (the Hugin "align" step)
  stitch.py      gain match, seam finder, optical-flow parallax pass, blending, JPEG/XMP output
  viewer.py      Tkinter 360° viewer widget
  app.py         the desktop app
  cli.py         command line
tests/           synthetic end-to-end tests
run_app.py / run_app.bat / run_app.sh   launchers
```
