"""insta360stitch - stitch and view Insta360 ONE X dual-fisheye .insp photos.

The pipeline is modelled on Hugin: a lens model seeded from the factory
calibration stored in the file, control points found automatically in the
overlap zones and optimised, then a seam finder and multi-band blender.  On
top of that an optical-flow pass warps the overlap zones so that parallax
between the two lenses is removed locally instead of ghosting.
"""

__version__ = "0.1.0"
