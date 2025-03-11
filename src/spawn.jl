# spawn.jl

"""
    spawner(fun!, nthread::Int, ntask::Int)
Apply `fun!(buffer_id, task)` for `task` in `1:ntask`
where `buffer_id` is in `{1, …, nthread}`

A single-thread version of this would simply be `fun!.(1, 1:ntask)`
"""
function spawner(fun!, nthread::Int, ntask::Int)
    chunks = Iterators.partition(1:ntask, ceil(Int, ntask/nthread)) # ≤ nthread chunks
    chunks = zip(1:nthread, chunks) # include a "task id" for buffer index
    tasks = map(chunks) do chunk
        Threads.@spawn fun!.(chunk...)
    end
    fetch.(tasks)
end
