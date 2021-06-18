# push!(LOAD_PATH,"../src/")

using SPECTrecon
using Documenter

DocMeta.setdocmeta!(SPECTrecon, :DocTestSetup, :(using SPECTrecon); recursive=true)

makedocs(;
    modules = [SPECTrecon],
    authors = "Jeff Fessler and contributors",
    repo = "https://github.com/JeffFessler/SPECTrecon.jl/blob/{commit}{path}#{line}",
    sitename = "SPECTrecon.jl",
    format = Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
#       canonical="https://JeffFessler.github.io/SPECTrecon.jl",
#       assets=String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JeffFessler/SPECTrecon.jl.git",
    devbranch = "main",
    devurl = "dev",
    versions = ["stable" => "v^", "dev" => "dev"]
#   push_preview = true,
)
