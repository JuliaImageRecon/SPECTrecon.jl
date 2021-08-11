fftshift!(dst, src) = circshift!(dst, src, div.([size(src)...],2))
ifftshift!(dst, src) = circshift!(dst, src, div.([size(src)...],-2))
function pad_it!(X,padsize)

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

function myfft!(img, tmp)
    ifftshift!(tmp, img)
    fft!(tmp)
    fftshift!(img, tmp)
end
function myifft!(img, tmp)
    ifftshift!(tmp, img)
    ifft!(tmp)
    fftshift!(img, tmp)
end
@btime pad_it!(y, (41, 60))
ker = 1/9 * ones(3,3)
img = zeros(16, 16)
img[4:13, 4:13] .= randn(10, 10)
tmp_compl = ones(ComplexF32, 16, 16)
ker_compl = ones(ComplexF32, 16, 16)
img_compl = ones(ComplexF32, 16, 16)
img_compl .= img .+ 0im
ker_compl .= pad_it!(ker, (16, 16)) .+ 0im
myfft!(img_compl, tmp_compl)
myfft!(ker_compl, tmp_compl)
img_compl .*= ker_compl
myifft!(img_compl, tmp_compl)

out = imfilter(img, centered(ker))


y = rand(16, 16)
x_compl = rand(ComplexF32, 16, 16)
y_compl = rand(ComplexF32, 16, 16)
@btime y_compl .= y .+ 0im
@btime fftshift(fft(ifftshift(y)))
@btime myfft!(y_compl, x_compl)
@btime y .= x .+ 0im
