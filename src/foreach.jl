

"""
     foreach_setup(z)
Return `Channel` for `foreach` threaded computation from iterable `z`.
"""
function foreach_setup(z)
    return Channel{Int}(length(z)) do ch
        foreach(i -> put!(ch, i), z)
    end
end
