# Syntax validation for EGARCH model script
# This checks if the file has valid Julia syntax without loading packages

println("Validating Julia syntax for egarch_model.jl...")
println()

# Read the file
file_path = "egarch_model.jl"
code = read(file_path, String)

# Try to parse the entire file
try
    # Parse the code
    expr = Meta.parse("begin\n$code\nend")

    println("✓ Syntax validation successful!")
    println()
    println("File statistics:")
    println("  - File size: $(length(code)) characters")
    println("  - Lines of code: $(count('\n', code) + 1)")

    # Count key elements
    n_functions = count(r"^function ", code, overlap=false) + count(r"^@model function ", code, overlap=false)
    n_exports = count(r"export ", code)

    println("  - Functions defined: $n_functions")
    println("  - Exported symbols: $n_exports")

    println()
    println("Main functions found:")
    for match in eachmatch(r"(?:^|\n)(?:@model )?function ([a-zA-Z_][a-zA-Z0-9_!]*)", code)
        println("  - $(match.captures[1])()")
    end

    println()
    println("The script has valid Julia syntax! ✓")
    println()
    println("Note: Package dependencies need to be installed to run:")
    println("  - Turing")
    println("  - Distributions")
    println("  - LinearAlgebra")
    println("  - Random")
    println("  - StatsBase")

catch e
    println("✗ Syntax error found:")
    println(e)
    exit(1)
end
