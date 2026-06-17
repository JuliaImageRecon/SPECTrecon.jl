# SPECTrecon
SPECT image reconstruction tools in Julia

https://github.com/JuliaImageRecon/SPECTrecon.jl

[![docs-stable][docs-stable-img]][docs-stable-url]
[![docs-dev][docs-dev-img]][docs-dev-url]
[![action][action-img]][action-url]
[![Aqua QA][aqua-img]][aqua-url]
[![codecov][codecov-img]][codecov-url]
[![deps][deps-img]][deps-url]
[![license][license-img]][license-url]
[![pkgeval][pkgeval-img]][pkgeval-url]
[![version][ver-img]][ver-url]

This repo provides forward and back-projection methods
for SPECT image reconstruction.

It also has methods for ML-EM and ML-OS-EM image reconstruction.

For examples with graphics,
see the
[documentation][docs-stable-url].
The examples include an illustration
of how to integrate deep learning
into SPECT reconstruction.

Tested with Julia ≥ 1.12.

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

[aqua-img]: https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[codecov-img]: https://codecov.io/github/JuliaImageRecon/SPECTrecon.jl/coverage.svg
[codecov-url]: https://codecov.io/github/JuliaImageRecon/SPECTrecon.jl

[deps-img]: https://juliahub.com/docs/SPECTrecon/deps.svg
[deps-url]: https://juliahub.com/ui/Packages/SPECTrecon

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://JuliaImageRecon.github.io/SPECTrecon.jl/dev
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaImageRecon.github.io/SPECTrecon.jl/stable

[license-img]: https://img.shields.io/badge/license-MIT-brightgreen.svg
[license-url]: LICENSE

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/S/SPECTrecon.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/S/SPECTrecon.html

[ver-img]: https://juliahub.com/docs/SPECTrecon/version.svg
[ver-url]: https://juliahub.com/ui/Packages/SPECTrecon
