```@meta
CurrentModule = SPECTrecon
```

# Documentation for [SPECTrecon](https://github.com/JeffFessler/SPECTrecon.jl)

## Overview

This Julia module provides
SPECT forward and back projectors
for parallel-beam collimators,
accounting for attenuation
and depth-dependent collimator response.
(Compton scatter within the object is not modeled.)

Designed for use with the Michigan Image Reconstruction Toolbox
[(MIRT)](https://github.com/JeffFessler/MIRT.jl)
and similar frameworks.

The method implemented here is based on the 1992 paper
by GL Zeng & GT Gullberg
"Frequency domain implementation
of the three-dimensional geometric point response correction in SPECT imaging"
(IEEE Tr. on Nuclear Science, 39(5-1):1444-53, Oct 1992)
[(DOI)](http://doi.org/10.1109/23.173222).

The forward projection method works as follows.
* The emission image and the attenuation map
  are rotated (slice by slice) to the desired view angle
  using either 2D bilinear interpolation
  or a 3-pass rotation method based on linear interpolation.
* Each (rotated) plane is convolved with the given
  point spread function (PSF) of the collimator
  and summed, accounting for attenuation
  using the "central ray" approximation.

The back-projection method is the exact adjoint
of the forward projector.

See the
[Examples](@ref 1-overview)
tab to the left for usage details.
