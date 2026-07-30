using Documenter
using GammaPeakFits

DocMeta.setdocmeta!(GammaPeakFits, :DocTestSetup, :(using GammaPeakFits); recursive = true)

makedocs(
    sitename = "GammaPeakFits",
    modules = [GammaPeakFits],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://rpretsch.github.io/GammaPeakFits.jl/stable/",
    ),
    pages = ["Home" => "index.md", "API" => "api.md"],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    warnonly = ("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/rpretsch/GammaPeakFits.jl.git",
    forcepush = true,
    push_preview = true,
)
