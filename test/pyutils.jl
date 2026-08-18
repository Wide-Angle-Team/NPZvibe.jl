# Locating a python with numpy, and running scripts against it.  Every test that
# needs numpy is skipped (loudly) when none is available.

function find_python()
    candidates = String[]
    haskey(ENV, "NPZVIBE_PYTHON") && push!(candidates, ENV["NPZVIBE_PYTHON"])
    append!(candidates, ["python3", "python"])
    for cand in candidates
        try
            success(`$cand -c "import numpy"`) && return cand
        catch
            continue
        end
    end
    nothing
end

const PYTHON = find_python()
const HAVE_NUMPY = PYTHON !== nothing

if HAVE_NUMPY
    ver = strip(read(`$PYTHON -c "import numpy, sys; print(sys.version.split()[0], numpy.__version__)"`, String))
    @info "NPZvibe: cross-checking against python $ver at $PYTHON"
else
    @warn "NPZvibe: no python3 with numpy found; numpy interoperability tests are skipped"
end

"Run a python script file, returning its stdout and raising with its stderr on failure."
function pyrun(script::AbstractString, args...)
    out, err = IOBuffer(), IOBuffer()
    cmd = `$PYTHON $script $(String[string(a) for a in args])`
    p = run(pipeline(ignorestatus(cmd); stdout=out, stderr=err))
    p.exitcode == 0 || error("python failed (exit $(p.exitcode)) running $script:\n" *
                             String(take!(err)))
    String(take!(out))
end

"Run inline python code, returning its stdout and raising with its stderr on failure."
function pyexec(code::AbstractString, args...)
    out, err = IOBuffer(), IOBuffer()
    cmd = `$PYTHON -c $code $(String[string(a) for a in args])`
    p = run(pipeline(ignorestatus(cmd); stdout=out, stderr=err))
    p.exitcode == 0 || error("python failed (exit $(p.exitcode)):\n" * String(take!(err)))
    String(take!(out))
end

const PYDIR = @__DIR__
