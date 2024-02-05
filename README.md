# SPECTrecon
SPECT image reconstruction tools in Julia

https://github.com/JuliaImageRecon/SPECTrecon.jl

[![action status][action-img]][action-url]
[![pkgeval status][pkgeval-img]][pkgeval-url]
[![codecov][codecov-img]][codecov-url]
[![license][license-img]][license-url]
[![docs-stable][docs-stable-img]][docs-stable-url]
[![docs-dev][docs-dev-img]][docs-dev-url]
[![code-style][code-blue-img]][code-blue-url]
[![ColPrac: Contributor's Guide][colprac-img]][colprac-url]

This repo provides forward and back-projection methods
for SPECT image reconstruction.

It also has methods for ML-EM and ML-OS-EM image reconstruction.

For examples with graphics,
see the
[documentation][docs-stable-url].
The examples include an illustration
of how to integrate deep learning
into SPECT reconstruction.

Tested with Julia ≥ 1.10.

## Related packages

- Pytorch/GPU version:
https://github.com/ZongyuLi-umich/SPECTrecon-pytorch

- Julia/GPU version:
[CuSPECTrecon.jl](https://github.com/JuliaImageRecon/CuSPECTrecon.jl).

Designed for use with the
[Michigan Image Reconstruction Toolbox (MIRT)](https://github.com/JeffFessler/MIRT.jl)
or similar frameworks.

<!-- URLs -->
[action-img]: https://github.com/JuliaImageRecon/SPECTrecon.jl/workflows/CI/badge.svg
[action-url]: https://github.com/JuliaImageRecon/SPECTrecon.jl/actions
[build-img]: https://github.com/JuliaImageRecon/SPECTrecon.jl/workflows/CI/badge.svg?branch=main
[build-url]: https://github.com/JuliaImageRecon/SPECTrecon.jl/actions?query=workflow%3ACI+branch%3Amain
[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/S/SPECTrecon.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/S/SPECTrecon.html
[code-blue-img]: https://img.shields.io/badge/code%20style-blue-4495d1.svg
[code-blue-url]: https://github.com/invenia/BlueStyle
[codecov-img]: https://codecov.io/github/JuliaImageRecon/SPECTrecon.jl/coverage.svg?branch=main
[codecov-url]: https://codecov.io/github/JuliaImageRecon/SPECTrecon.jl?branch=main
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaImageRecon.github.io/SPECTrecon.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://JuliaImageRecon.github.io/SPECTrecon.jl/dev
[license-img]: https://img.shields.io/badge/license-MIT-brightgreen.svg
[license-url]: LICENSE
[colprac-img]: https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet
[colprac-url]: https://github.com/SciML/ColPrac
