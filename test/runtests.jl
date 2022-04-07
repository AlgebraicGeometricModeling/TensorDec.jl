for f in readdir(pwd())
    if startswith(f,"test")
        @info "Reading "*f
        try
            include(f)
        catch
            @error "Problem with "*f
            continue 
        end
    end
end

