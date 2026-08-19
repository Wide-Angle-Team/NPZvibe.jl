# Locating a python with numpy, and running scripts against it.  Tests that need
# numpy run by default and will fail when numpy is unavailable.  Set
# NPZVIBE_TEST_NUMPY=0 to skip them instead (opposite of NPZVIBE_TEST_LARGE).

const TEST_NUMPY = get(ENV, "NPZVIBE_TEST_NUMPY", "1") != "0"

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

if !TEST_NUMPY
    @info "NPZvibe: numpy interoperability tests skipped; set NPZVIBE_TEST_NUMPY=1 to run them"
elseif HAVE_NUMPY
    ver = strip(read(`$PYTHON -c "import numpy, sys; print(sys.version.split()[0], numpy.__version__)"`, String))
    @info "NPZvibe: cross-checking against python $ver at $PYTHON"
else
    @warn "NPZvibe: no python3 with numpy found; numpy tests will fail (set NPZVIBE_TEST_NUMPY=0 to skip)"
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
