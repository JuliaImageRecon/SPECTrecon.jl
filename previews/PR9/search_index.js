var documenterSearchIndex = {"docs":
[{"location":"methods/#Methods-list","page":"Methods","title":"Methods list","text":"","category":"section"},{"location":"methods/","page":"Methods","title":"Methods","text":"","category":"page"},{"location":"methods/#Methods-usage","page":"Methods","title":"Methods usage","text":"","category":"section"},{"location":"methods/","page":"Methods","title":"Methods","text":"Modules = [SPECTrecon]","category":"page"},{"location":"methods/#SPECTrecon.SPECTrecon","page":"Methods","title":"SPECTrecon.SPECTrecon","text":"SPECTrecon\n\nSystem matrix (forward and back-projector) for SPECT image reconstruction.\n\n\n\n\n\n","category":"module"},{"location":"methods/#SPECTrecon.SPECTplan","page":"Methods","title":"SPECTrecon.SPECTplan","text":"SPECTplan\n\nStruct for storing key factors for a SPECT system model\n\nT datatype of work arrays\nimgr [nx, ny, nz] 3D rotated version of image\nadd_img [nx, ny, nz] 3D image for adding views and backprojection\nmumap [nx,ny,nz] attenuation map, must be 3D, possibly zeros()\nmumapr [nx, ny, nz] 3D rotated mumap\npsfs [nx_psf,nz_psf,ny,nview] point spread function, must be 4D, with nx_psf and nz_psf odd, and symmetric for each slice\nnview number of views, must be integer\nviewangle set of view angles, must be from 0 to 2π\ninterpidx interpolation method, 1 means 1d, 2 means 2d\ndy voxel size in y direction (dx is the same value)\nimgsize{nx, ny, nz} number of voxels in {x,y,z} direction of the image, must be integer\npsfsize{nx_psf, nz_psf} number of voxels in {x, z} direction of the psf, must be integer\npad_fft{padu_fft, pad_fft, padl_fft, padr_fft} pixels padded for {left,right,up,down} direction for convolution with psfs, must be integer\npad_rot{padu_rot, padd_rot, padl_rot, padr_rot} padded pixels for {left,right,up,down} direction for image rotation\nncore number of CPU cores used to process data, must be integer\n\nCurrently code assumes the following:\n\neach of the nview projection views is [nx,nz]\nnx = ny\nuniform angular sampling\npsf is symmetric\nmultiprocessing using # of cores specified by Threads.nthreads()\n\n\n\n\n\n","category":"type"},{"location":"methods/#SPECTrecon.SPECTplan-Tuple{}","page":"Methods","title":"SPECTrecon.SPECTplan","text":"SPECTplan(mumap, psfs, dy; viewangle, interpidx, T)\n\nConstructor for SPECTplan\n\nIn\n\nmumap::AbstractArray{<:RealU, 3} 3D attenuation map\npsfs::AbstractArray{<:RealU, 4} 4D PSF array\ndy::RealU pixel size\n\nOption\n\nviewangle::AbstractVector{<:RealU} default 0 to almost 2π\ninterpidx::Int = 2 1 is for 3-pass 1D interpolator; 2 is for 2D interpolator\nT::DataType = promote_type(eltype(mumap), Float32)\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.Workarray","page":"Methods","title":"SPECTrecon.Workarray","text":"Workarray\n\nStruct for storing keys of the work array for a single thread add tmp vectors to avoid allocating in rotatex and rotatey For fft convolution:\n\nworkmat_fft [nx+padu_fft+padd_fft, nz+padl_fft+padr_fft]: 2D padded image for imfilter3\nworkvec_fft_1 [nz+padl_fft+padr_fft,]: 1D work vector\nworkvec_fft_2 [nx+padu_fft+padd_fft,]: 1D work vector\nimg_compl [nx+padu_fft+padd_fft, nz+padl_fft+padr_fft]: 2D [complex] padded image for fft\nker_compl [nx+padu_fft+padd_fft, nz+padl_fft+padr_fft]: 2D [complex] padded image for fft\nfft_plan plan for doing fft, see plan_fft!\nifft_plan plan for doing ifft, see plan_ifft!\n\nFor image rotation:\n\nworkmat_rot_1 [nx+padu_rot+padd_rot, ny+padl_rot+padr_rot]: 2D padded image for image rotation\nworkmat_rot_2 [nx+padu_rot+padd_rot, ny+padl_rot+padr_rot]: 2D padded image for image rotation\nworkvec_rot_x [nx+padu_rot+padd_rot,]: 1D work vector for image rotation\nworkvec_rot_y [ny+padl_rot+padr_rot,]: 1D work vector for image rotation\ninterp_x sparse interpolator for rotating in x direction\ninterp_y sparse interpolator for rotating in y direction\n\nFor attenuation:\n\nexp_mumapr [nx, nz] 2D exponential rotated mumap\n\n\n\n\n\n","category":"type"},{"location":"methods/#SPECTrecon.backproject!-Tuple{AbstractArray{var\"#s49\", 3} where var\"#s49\"<:Number, AbstractArray{var\"#s30\", 3} where var\"#s30\"<:Number, SPECTplan, Vector{Workarray}}","page":"Methods","title":"SPECTrecon.backproject!","text":"backproject!(image, views, plan, workarray; index)\n\nBackproject multiple views into image. Array image is not initialized to zero; caller must do that.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.backproject!-Tuple{AbstractArray{var\"#s51\", 3} where var\"#s51\"<:Number, AbstractMatrix{var\"#s50\"} where var\"#s50\"<:Number, SPECTplan, Vector{Workarray}, Int64}","page":"Methods","title":"SPECTrecon.backproject!","text":"backproject!(image, view, plan, workarray, viewidx)\n\nBackproject a single view.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.backproject-Tuple{AbstractArray{var\"#s49\", 3} where var\"#s49\"<:Number, AbstractArray{var\"#s30\", 3} where var\"#s30\"<:Number, AbstractArray{var\"#s19\", 4} where var\"#s19\"<:Number, Number}","page":"Methods","title":"SPECTrecon.backproject","text":"image = backproject(views, mumap, psfs, dy; interpidx, kwargs...)\n\nSPECT backproject views using attenuation map mumap and PSF array psfs for pixel size dy. This method initializes the plan and workarray as a convenience. Most users should use backproject! instead after initializing those, for better efficiency.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.backproject-Tuple{SPECTplan, Vector{Workarray}, AbstractArray{var\"#s59\", 3} where var\"#s59\"<:Number}","page":"Methods","title":"SPECTrecon.backproject","text":"image = backproject(plan, workarray, views ; kwargs...)\n\nSPECT backproject views; this allocates the returned 3D array.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.copy3dj!-Tuple{AbstractMatrix{T} where T, AbstractArray{var\"#s17\", 3} where var\"#s17\", Int64}","page":"Methods","title":"SPECTrecon.copy3dj!","text":"copy3dj!(mat2d, mat3d, j)\n\nNon-allocating mat2d .= mat3d[:,j,:]\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.fft_conv!-Tuple{AbstractMatrix{var\"#s50\"} where var\"#s50\"<:Number, AbstractMatrix{var\"#s49\"} where var\"#s49\"<:Number, AbstractMatrix{var\"#s30\"} where var\"#s30\"<:Number, AbstractMatrix{var\"#s19\"} where var\"#s19\"<:Number, NTuple{4, Int64}, AbstractMatrix{var\"#s59\"} where {T<:AbstractFloat, var\"#s59\"<:Complex{T}}, AbstractMatrix{var\"#s59\"} where {T<:AbstractFloat, var\"#s59\"<:Complex{T}}, Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}, Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}}","page":"Methods","title":"SPECTrecon.fft_conv!","text":"fft_conv!(output, workmat, img, ker, fftpadsize, img_compl, ker_compl, fft_plan, ifft_plan)\n\nConvolve img with ker using FFT, and store the result in output\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.fft_conv_adj!-Union{Tuple{T}, Tuple{AbstractMatrix{var\"#s18\"} where var\"#s18\"<:Number, AbstractMatrix{var\"#s17\"} where var\"#s17\"<:Number, AbstractVector{T}, AbstractVector{T}, AbstractMatrix{var\"#s11\"} where var\"#s11\"<:Number, AbstractMatrix{var\"#s6\"} where var\"#s6\"<:Number, NTuple{4, Int64}, AbstractMatrix{var\"#s59\"} where {T<:AbstractFloat, var\"#s59\"<:Complex{T}}, AbstractMatrix{var\"#s59\"} where {T<:AbstractFloat, var\"#s59\"<:Complex{T}}, Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}, Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}}} where T","page":"Methods","title":"SPECTrecon.fft_conv_adj!","text":"fft_conv_adj!(output, workmat, workvec1, workvec2, img, ker,\n              fftpadsize, img_compl, ker_compl, fft_plan, ifft_plan)\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.fftshift2!-Tuple{AbstractMatrix{T} where T, AbstractMatrix{T} where T}","page":"Methods","title":"SPECTrecon.fftshift2!","text":"fftshift2!(dst, src)\n\nSame as fftshift in 2d, but non-allocating\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.imfilter3!-Tuple{AbstractMatrix{var\"#s57\"} where var\"#s57\"<:Number, AbstractMatrix{var\"#s59\"} where {T<:AbstractFloat, var\"#s59\"<:Complex{T}}, AbstractMatrix{var\"#s53\"} where var\"#s53\"<:Number, AbstractMatrix{var\"#s59\"} where {T<:AbstractFloat, var\"#s59\"<:Complex{T}}, Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}, Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}}","page":"Methods","title":"SPECTrecon.imfilter3!","text":"imfilter3!(output, img_compl, ker, ker_compl, fft_plan, ifft_plan)\n\nFFT-based convolution between img_compl and kernel ker (not centered) putting result in output.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.imfilter3_adj!-Tuple{AbstractMatrix{var\"#s57\"} where var\"#s57\"<:Number, AbstractMatrix{var\"#s59\"} where {T<:AbstractFloat, var\"#s59\"<:Complex{T}}, AbstractMatrix{var\"#s53\"} where var\"#s53\"<:Number, AbstractMatrix{var\"#s59\"} where {T<:AbstractFloat, var\"#s59\"<:Complex{T}}, Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}, Union{AbstractFFTs.ScaledPlan, FFTW.cFFTWPlan}}","page":"Methods","title":"SPECTrecon.imfilter3_adj!","text":"imfilter3_adj!(output, img_compl, kerev, ker_compl, fft_plan, ifft_plan)\n\nApply FFT convolution between img_compl and REVERSED kernel (not centered), assuming the kernel is already be in reversed order.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.imrotate3!-Tuple{AbstractMatrix{var\"#s17\"} where var\"#s17\"<:Number, AbstractMatrix{var\"#s11\"} where var\"#s11\"<:Number, AbstractMatrix{var\"#s6\"} where var\"#s6\"<:Number, AbstractMatrix{var\"#s5\"} where var\"#s5\"<:Number, Real, LinearInterpolators.SparseInterpolators.SparseInterpolator{var\"#s4\", S, N} where {var\"#s4\"<:AbstractFloat, S, N}, LinearInterpolators.SparseInterpolators.SparseInterpolator{var\"#s3\", S, N} where {var\"#s3\"<:AbstractFloat, S, N}, AbstractVector{var\"#s2\"} where var\"#s2\"<:Number, AbstractVector{var\"#s57\"} where var\"#s57\"<:Number}","page":"Methods","title":"SPECTrecon.imrotate3!","text":"imrotate3!(output, workmat1, workmat2, img, θ, interp_x, interp_y, workvec_x, workvec_y)\n\nRotate an image by angle θ (must be within 0 to 2π) in clockwise direction using a series of 1d linear interpolations.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.imrotate3!-Tuple{AbstractMatrix{var\"#s51\"} where var\"#s51\"<:Number, AbstractMatrix{var\"#s50\"} where var\"#s50\"<:Number, AbstractMatrix{var\"#s49\"} where var\"#s49\"<:Number, AbstractMatrix{var\"#s30\"} where var\"#s30\"<:Number, Real}","page":"Methods","title":"SPECTrecon.imrotate3!","text":"imrotate3!(output, workmat1, workmat2, img, θ)\n\nRotate an image by angle θ in clockwise direction using 2d linear interpolation Source code is here: https://github.com/emmt/LinearInterpolators.jl\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.imrotate3_adj!-Tuple{AbstractMatrix{var\"#s18\"} where var\"#s18\"<:Number, AbstractMatrix{var\"#s17\"} where var\"#s17\"<:Number, AbstractMatrix{var\"#s11\"} where var\"#s11\"<:Number, AbstractMatrix{var\"#s6\"} where var\"#s6\"<:Number, Real, LinearInterpolators.SparseInterpolators.SparseInterpolator{var\"#s5\", S, N} where {var\"#s5\"<:AbstractFloat, S, N}, LinearInterpolators.SparseInterpolators.SparseInterpolator{var\"#s4\", S, N} where {var\"#s4\"<:AbstractFloat, S, N}, AbstractVector{var\"#s3\"} where var\"#s3\"<:Number, AbstractVector{var\"#s2\"} where var\"#s2\"<:Number}","page":"Methods","title":"SPECTrecon.imrotate3_adj!","text":"imrotate3_adj!(output, workmat1, workmat2, img, θ, interp_x, interp_y, workvec_x, workvec_y)\n\nThe adjoint of rotating an image by angle θ (must be ranging from 0 to 2π) in clockwise direction using a series of 1d linear interpolation\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.imrotate3_adj!-Tuple{AbstractMatrix{var\"#s51\"} where var\"#s51\"<:Number, AbstractMatrix{var\"#s50\"} where var\"#s50\"<:Number, AbstractMatrix{var\"#s49\"} where var\"#s49\"<:Number, AbstractMatrix{var\"#s30\"} where var\"#s30\"<:Number, Real}","page":"Methods","title":"SPECTrecon.imrotate3_adj!","text":"imrotate3_adj!(output, workmat1, workmat2, img, θ)\n\nThe adjoint of rotating an image by angle θ in clockwise direction using 2d linear interpolation Source code is here: https://github.com/emmt/LinearInterpolators.jl\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.linearinterp!-Tuple{LinearInterpolators.SparseInterpolators.SparseInterpolator{var\"#s6\", S, N} where {var\"#s6\"<:AbstractFloat, S, N}, AbstractVector{var\"#s5\"} where var\"#s5\"<:Number}","page":"Methods","title":"SPECTrecon.linearinterp!","text":"linearinterp!(A, x)\n\nAssign key values in SparseInterpolator (linear) A that are calculated from x. x must be a constant vector\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.mul3dj!-Tuple{AbstractArray{var\"#s17\", 3} where var\"#s17\", AbstractMatrix{T} where T, Int64}","page":"Methods","title":"SPECTrecon.mul3dj!","text":"mul3dj!(mat3d, mat2d, j)\n\nNon-allocating mat3d[:,j,:] *= mat2d\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.pad2sizezero!-Union{Tuple{T}, Tuple{AbstractMatrix{T}, AbstractMatrix{T} where T, Tuple{Int64, Int64}}} where T","page":"Methods","title":"SPECTrecon.pad2sizezero!","text":"pad2sizezero!(output, img, padsize)\n\nNon-allocating version of padding: `output[paddims[1]+1 : paddims[1]+dims[1],         paddims[2]+1 : paddims[2]+dims[2]]) .= img\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.pad_it!-Union{Tuple{T}, Tuple{D}, Tuple{AbstractArray{T, D}, Tuple{Vararg{Int64, D}}}} where {D, T<:Number}","page":"Methods","title":"SPECTrecon.pad_it!","text":"pad_it!(X, padsize)\n\nZero-pad X to padsize\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.padrepl!-Tuple{AbstractMatrix{T} where T, AbstractMatrix{T} where T, NTuple{4, Int64}}","page":"Methods","title":"SPECTrecon.padrepl!","text":"padrepl!(output, img, padsize)\n\nPad with replication from img into output\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.padzero!-Union{Tuple{T}, Tuple{AbstractMatrix{T}, AbstractMatrix{T} where T, NTuple{4, Int64}}} where T","page":"Methods","title":"SPECTrecon.padzero!","text":"padzero!(output, img, pad_x, pad_y)\n\nMutating version of padding a 2D image by filling zeros. Output has size (size(img, 1) + padsize[1] + padsize[2], size(img, 2) + padsize[3] + padsize[4]).\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.plus1di!-Tuple{AbstractMatrix{T} where T, AbstractVector{T} where T, Int64}","page":"Methods","title":"SPECTrecon.plus1di!","text":"plus1di!(mat2d, mat1d)\n\nNon-allocating mat2d[i, :] += mat1d\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.plus1dj!-Tuple{AbstractMatrix{T} where T, AbstractVector{T} where T, Int64}","page":"Methods","title":"SPECTrecon.plus1dj!","text":"plus1dj!(mat2d, mat1d)\n\nNon-allocating mat2d[:, j] += mat1d\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.plus2di!-Tuple{AbstractVector{T} where T, AbstractMatrix{T} where T, Int64}","page":"Methods","title":"SPECTrecon.plus2di!","text":"plus2di!(mat1d, mat2d, i)\n\nNon-allocating mat1d += mat2d[i,:]\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.plus2dj!-Tuple{AbstractVector{T} where T, AbstractMatrix{T} where T, Int64}","page":"Methods","title":"SPECTrecon.plus2dj!","text":"plus2dj!(mat1d, mat2d, j)\n\nNon-allocating mat1d += mat2d[:,j]\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.plus3di!-Tuple{AbstractMatrix{T} where T, AbstractArray{var\"#s17\", 3} where var\"#s17\", Int64}","page":"Methods","title":"SPECTrecon.plus3di!","text":"plus3di!(mat2d, mat3d, i)\n\nNon-allocating mat2d += mat3d[i,:,:]\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.plus3dj!-Tuple{AbstractMatrix{T} where T, AbstractArray{var\"#s17\", 3} where var\"#s17\", Int64}","page":"Methods","title":"SPECTrecon.plus3dj!","text":"plus3dj!(mat2d, mat3d, j)\n\nNon-allocating mat2d += mat3d[:,j,:]\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.plus3dk!-Tuple{AbstractMatrix{T} where T, AbstractArray{var\"#s17\", 3} where var\"#s17\", Int64}","page":"Methods","title":"SPECTrecon.plus3dk!","text":"plus3dk!(mat2d, mat3d, k)\n\nNon-allocating mat2d += mat3d[:,:,k]\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.project!-Tuple{AbstractArray{var\"#s49\", 3} where var\"#s49\"<:Number, AbstractArray{var\"#s30\", 3} where var\"#s30\"<:Number, SPECTplan, Vector{Workarray}}","page":"Methods","title":"SPECTrecon.project!","text":"project!(views, image, plan, workarray; index)\n\nProject image into multiple views with indexes index (defaults to 1:nview). The 3D views array must be pre-allocated, but need not be initialized.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.project!-Tuple{AbstractMatrix{var\"#s52\"} where var\"#s52\"<:Number, AbstractArray{var\"#s51\", 3} where var\"#s51\"<:Number, SPECTplan, Vector{Workarray}, Int64}","page":"Methods","title":"SPECTrecon.project!","text":"project!(view, plan, workarray, image, viewidx)\n\nSPECT projection of image into a single view with index viewidx. The view must be pre-allocated but need not be initialized to zero.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.project-Tuple{AbstractArray{var\"#s49\", 3} where var\"#s49\"<:Number, AbstractArray{var\"#s30\", 3} where var\"#s30\"<:Number, AbstractArray{var\"#s19\", 4} where var\"#s19\"<:Number, Number}","page":"Methods","title":"SPECTrecon.project","text":"views = project(image, mumap, psfs, dy; interpidx, kwargs...)\n\nConvenience method for SPECT forward projector that does all allocation including initializing plan and workarray.\n\nIn\n\nimage : 3D array [nx,ny,nz]\nmumap : [nx,ny,nz] 3D attenuation map, possibly zeros()\npsfs : 4D PSF array\ndy::RealU : pixel size\n\nOption\n\ninterpidx : 1 or 2\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.project-Tuple{AbstractArray{var\"#s59\", 3} where var\"#s59\"<:Number, SPECTplan, Vector{Workarray}}","page":"Methods","title":"SPECTrecon.project","text":"views = project(image, plan, workarray ; kwargs...)\n\nConvenience method for SPECT forward projector that allocates and returns views.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.rot180!-Tuple{AbstractMatrix{T} where T, AbstractMatrix{T} where T}","page":"Methods","title":"SPECTrecon.rot180!","text":"rot180!(B::AbstractMatrix, A::AbstractMatrix)\n\nIn place version of rot180, returning rotation of A in B.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.rot_f90!-Tuple{AbstractMatrix{T} where T, AbstractMatrix{T} where T, Int64}","page":"Methods","title":"SPECTrecon.rot_f90!","text":"rot_f90!(output, img, m)\n\nIn-place version of rotating an image by 90/180/270 degrees\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.rot_f90_adj!-Tuple{AbstractMatrix{T} where T, AbstractMatrix{T} where T, Int64}","page":"Methods","title":"SPECTrecon.rot_f90_adj!","text":"rot_f90_adj!(output, img, m)\nThe adjoint of rotating an image by 90/180/270 degrees\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.rotate_x!-Tuple{AbstractMatrix{var\"#s4\"} where var\"#s4\"<:Number, AbstractMatrix{var\"#s3\"} where var\"#s3\"<:Number, Real, AbstractVector{var\"#s2\"} where var\"#s2\"<:Number, AbstractVector{var\"#s49\"} where var\"#s49\"<:Number, LinearInterpolators.SparseInterpolators.SparseInterpolator{var\"#s50\", S, N} where {var\"#s50\"<:AbstractFloat, S, N}, AbstractVector{var\"#s51\"} where var\"#s51\"<:Number, Real}","page":"Methods","title":"SPECTrecon.rotate_x!","text":"rotate_x!(output, img, tan_θ, xi, yi, interp, workvec, c_y)\n\nRotate a 2D image along x axis in clockwise direction using 1d linear interpolation, storing results in output. Sample locations xi and yi must be in increasing order.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.rotate_x_adj!-Tuple{AbstractMatrix{var\"#s19\"} where var\"#s19\"<:Number, AbstractMatrix{var\"#s18\"} where var\"#s18\"<:Number, Real, AbstractVector{var\"#s17\"} where var\"#s17\"<:Number, AbstractVector{var\"#s11\"} where var\"#s11\"<:Number, LinearInterpolators.SparseInterpolators.SparseInterpolator, AbstractVector{var\"#s6\"} where var\"#s6\"<:Number, Real}","page":"Methods","title":"SPECTrecon.rotate_x_adj!","text":"rotate_x_adj!(output, img, tan_θ, xi, yi, interp, workvec, c_y)\n\nThe adjoint of rotating a 2D image along x axis in clockwise direction using 1d linear interpolation, storing results in output xi and yi must be in increasing order.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.rotate_y!-Tuple{AbstractMatrix{var\"#s19\"} where var\"#s19\"<:Number, AbstractMatrix{var\"#s18\"} where var\"#s18\"<:Number, Real, AbstractVector{var\"#s17\"} where var\"#s17\"<:Number, AbstractVector{var\"#s11\"} where var\"#s11\"<:Number, LinearInterpolators.SparseInterpolators.SparseInterpolator, AbstractVector{var\"#s6\"} where var\"#s6\"<:Number, Real}","page":"Methods","title":"SPECTrecon.rotate_y!","text":"rotate_y!(output, img, sin_θ, xi, yi, interp, workvec, c_x)\n\nRotate a 2D image along y axis in clockwise direction using 1d linear interpolation, storing results in output xi and yi must be in increasing order.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.rotate_y_adj!-Tuple{AbstractMatrix{var\"#s19\"} where var\"#s19\"<:Number, AbstractMatrix{var\"#s18\"} where var\"#s18\"<:Number, Real, AbstractVector{var\"#s17\"} where var\"#s17\"<:Number, AbstractVector{var\"#s11\"} where var\"#s11\"<:Number, LinearInterpolators.SparseInterpolators.SparseInterpolator, AbstractVector{var\"#s6\"} where var\"#s6\"<:Number, Real}","page":"Methods","title":"SPECTrecon.rotate_y_adj!","text":"rotate_y_adj!(output, img, sin_θ, xi, yi, interp, workvec, c_x)\n\nAdjoint of rotating a 2D image along y axis in clockwise direction using 1d linear interpolation, storing results in output\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.rotl90!-Tuple{AbstractMatrix{T} where T, AbstractMatrix{T} where T}","page":"Methods","title":"SPECTrecon.rotl90!","text":"rotl90!(B::AbstractMatrix, A::AbstractMatrix)\n\nIn place version of rotl90, returning rotation of A in B.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.rotr90!-Tuple{AbstractMatrix{T} where T, AbstractMatrix{T} where T}","page":"Methods","title":"SPECTrecon.rotr90!","text":"rotr90!(B::AbstractMatrix, A::AbstractMatrix)\n\nIn place version of rotr90, returning rotation of A in B.\n\n\n\n\n\n","category":"method"},{"location":"methods/#SPECTrecon.scale3dj!-Tuple{AbstractMatrix{T} where T, AbstractArray{var\"#s17\", 3} where var\"#s17\", Int64, Number}","page":"Methods","title":"SPECTrecon.scale3dj!","text":"scale3dj!(mat2d, mat3d, j, s)\n\nNon-allocating mat2d = s * mat3d[:,j,:]\n\n\n\n\n\n","category":"method"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"EditURL = \"https://github.com/JeffFessler/SPECTrecon.jl/blob/master/docs/lit/examples/1-overview.jl\"","category":"page"},{"location":"examples/1-overview/#overview","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"","category":"section"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"This page explains the Julia package SPECTrecon.","category":"page"},{"location":"examples/1-overview/#Setup","page":"SPECTrecon overview","title":"Setup","text":"","category":"section"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"Packages needed here.","category":"page"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"using SPECTrecon\nusing MIRTjim: jim, prompt\nusing Plots: scatter, plot!, default; default(markerstrokecolor=:auto)","category":"page"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"The following line is helpful when running this example.jl file as a script; this way it will prompt user to hit a key after each figure is displayed.","category":"page"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"isinteractive() ? jim(:prompt, true) : prompt(:draw);\nnothing #hide","category":"page"},{"location":"examples/1-overview/#Overview","page":"SPECTrecon overview","title":"Overview","text":"","category":"section"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"To perform SPECT image reconstruction, one must have a model for the imaging system encapsulated in a forward project and back projector.","category":"page"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"More details todo","category":"page"},{"location":"examples/1-overview/#Units","page":"SPECTrecon overview","title":"Units","text":"","category":"section"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"The pixel dimensions deltas can (and should!) be values with units.","category":"page"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"Here is an example ...","category":"page"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"using UnitfulRecipes\nusing Unitful: mm","category":"page"},{"location":"examples/1-overview/#Methods","page":"SPECTrecon overview","title":"Methods","text":"","category":"section"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"todo","category":"page"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"","category":"page"},{"location":"examples/1-overview/","page":"SPECTrecon overview","title":"SPECTrecon overview","text":"This page was generated using Literate.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SPECTrecon","category":"page"},{"location":"#Documentation-for-[SPECTrecon](https://github.com/JeffFessler/SPECTrecon.jl)","page":"Home","title":"Documentation for SPECTrecon","text":"","category":"section"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This Julia module provides SPECT forward and back projectors for parallel-beam collimators. Designed for use with the Michigan Image Reconstruction Toolbox (MIRT) and similar frameworks.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The method implemented here is based on the 1992 paper by GL Zeng & GT Gullberg \"Frequency domain implementation of the three-dimensional geometric point response correction in SPECT imaging\" (DOI).","category":"page"},{"location":"","page":"Home","title":"Home","text":"See the Examples tab to the left for usage details.","category":"page"}]
}
