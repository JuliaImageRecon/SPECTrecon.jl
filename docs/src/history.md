# History

The SPECT forward and back-projection method implemented here
is based on the 1992 paper
by GL Zeng & GT Gullberg
"Frequency domain implementation
of the three-dimensional geometric point response correction in SPECT imaging"
[(DOI)](http://doi.org/10.1109/23.173222).


Historically
there have been relatively few
open-source libraries
for SPECT image reconstruction,
and to the best of our knowledge
source code for the Zeng & Gullberg approach
has not been available prior to this package.

Starting in about 1995,
the
[ASPIRE](https://web.eecs.umich.edu/~fessler/aspire/index.html)
library,
developed in the early 1990s
at the University of Michigan,
began providing precompiled binaries
(from C99 source code)
for 2D SPECT image reconstruction
with
[documentation](https://web.eecs.umich.edu/~fessler/papers/files/tr/95,293,aspire3.pdf).

In about 1997,
a 3D version of ASPIRE
for SPECT reconstruction
became available,
again as
[precompiled binaries](https://web.eecs.umich.edu/~fessler/aspire/index.html)
with
[documentation](https://web.eecs.umich.edu/~fessler/papers/files/tr/97,310,ugf.pdf).
Anastasia Yendiki
was a key contributor
to the SPECT code.
As noted in a
[2001 technical report](https://web.eecs.umich.edu/~fessler/papers/files/tr/spect3.pdf)
we took pains to ensure
that the forward and back-projector
were (adjoint) consistent pairs.
Around 2001
the work was extended
to consider
blob basis functions,
leading to a
[2004 comparison paper](http://doi.org/10.1088/0031-9155/49/11/003).

Somewhere during that period
the 3D SPECT projector / backprojector
became available
as precompiled MEX files
for use with the
[Matlab version of the Michigan Image Reconstruction Toolbox](https://github.com/JeffFessler/mirt).

The 3D version in ASPIRE
precomputes rotated versions of the attenuation map,
to save computation
at the price of substantially more memory.
That trade-off was reasonable
in the era before machine learning.
Today,
with a focus on end-to-end training
of image reconstruction methods
in all modalities,
including SPECT,
it is desirable
to have methods
that use less memory
to facilitate
GPU implementations.
This open-source Julia package
is designed for the machine learning era. 

Development work on this package
is supported in part by the following projects
led by Dr. Yuni Dewaraja:
* NIH Grant [R01 EB022075](https://grantome.com/grant/NIH/R01-EB022075-01A1)
* NIH Grant [R01 CA240706](https://grantome.com/grant/NIH/R01-CA240706-01A1)
