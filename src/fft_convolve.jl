# fft_convolve.jl
# up to 8x faster and more efficient than ImageFiltering.imfilter!
fftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, div.([size(src)...],2))
ifftshift!(dst::AbstractArray, src::AbstractArray) = circshift!(dst, src, div.([size(src)...],-2))
function pad_it!(X::AbstractArray,
                padsize::Tuple)

    pad1,pad2 = padsize
    M, N = size(X)
    # Fill(x,(a,b),(c,d)) adds 'a' rows, 'b' columns before x;
    # and 'c' rows, 'd' columns after x.
    Xpad = OffsetArrays.no_offset_view(BorderArray(X,
                      Fill(0,
                      (ceil(Int, (pad1-M)/2), ceil(Int, (pad2-N)/2)),
                      (floor(Int, (pad1-M)/2), floor(Int, (pad2-N)/2)))))
    return Xpad
end

function imfilter3!(padimg::AbstractArray{<:Float32, 2},
                   ker::AbstractArray{<:Float32, 2},
                   img_compl::AbstractArray{ComplexF32, 2},
                   ker_compl::AbstractArray{ComplexF32, 2})

    img_compl .= padimg
    ker_compl .= pad_it!(ker, size(img_compl))
    fft!(img_compl)
    fft!(ker_compl)
    img_compl .*= ker_compl
    ifft!(img_compl)
    padimg .= real.(fftshift!(ker_compl, img_compl))
end

# @btime pad_it!(y, (41, 60))
# ker = convert(Matrix{Float32}, 1/9 * ones(Float32, 3,3))
# img = zeros(Float32, 16, 16)
# img[4:13, 4:13] .= randn(Float32, 10, 10)
# img_compl = ones(ComplexF32, 22, 22)
# ker_compl = similar(img_compl)
# tmp_compl = similar(img_compl)
# output1 = similar(img_compl, Float32)
# output2 = similar(img_compl, Float32)
# copyto!(output1, OffsetArrays.no_offset_view(BorderArray(img, Pad(:replicate, (3,3),(3,3)))))
# @btime imfilter!(output1, ker, img_compl, ker_compl, tmp_compl)
# ImageFiltering.imfilter!(output2, OffsetArrays.no_offset_view(BorderArray(
#                 img, Pad(:replicate, (3,3),(3,3)))), ker, NoPad(), Algorithm.FFT())
# ImageFiltering.imfilter!(output1, BorderArray(
#                 img, Pad(:replicate, (3,3),(3,3))), ker, NoPad(), Algorithm.FFT())


# myfft!(img_compl, tmp_compl)
# myfft!(ker_compl, tmp_compl)
# img_compl .*= ker_compl
# myifft!(img_compl, tmp_compl)
#
# out = imfilter(img, centered(ker))
#
#
# y = rand(16, 16)
# x_compl = rand(ComplexF32, 16, 16)
# y_compl = rand(ComplexF32, 16, 16)
# @btime y_compl .= y .+ 0im
# @btime fftshift(fft(ifftshift(y)))
# @btime myfft!(y_compl, x_compl)
# @btime y .= x .+ 0im
