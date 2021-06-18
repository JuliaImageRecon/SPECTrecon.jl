# push!(LOAD_PATH,"../src/")

using SPECT
using Documenter

DocMeta.setdocmeta!(SPECT, :DocTestSetup, :(using SPECT); recursive=true)

makedocs(;
    modules = [SPECT],
    authors = "Jeff Fessler and contributors",
    repo = "https://github.com/JeffFessler/SPECT.jl/blob/{commit}{path}#{line}",
    sitename = "SPECT.jl",
    format = Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
#       canonical="https://JeffFessler.github.io/SPECT.jl",
#       assets=String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JeffFessler/SPECT.jl.git",
    devbranch = "main",
    devurl = "dev",
    versions = ["stable" => "v^", "dev" => "dev"]
#   push_preview = true,
)
