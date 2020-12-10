# adapted from https://github.com/JuliaLang/www.julialang.org/blob/master/utils.jl
"""
    {{blogposts}}
Plug in the list of blog posts contained in the `/blog/` folder.
"""
function hfun_blogposts()
    curyear = year(Dates.today())
    io = IOBuffer()
    for year in curyear:-1:2020
        ys = "$year"
        year < curyear && write(io, "\n**$year**\n")
        for month in 12:-1:1
            ms = "0"^(month < 10) * "$month"
            base = joinpath("blog", ys, ms)
            isdir(base) || continue
            posts = filter!(p -> endswith(p, ".md"), readdir(base))
            days  = zeros(Int, length(posts))
            lines = Vector{String}(undef, length(posts))
            for (i, post) in enumerate(posts)
                ps  = splitext(post)[1]
                url = "/blog/$ys/$ms/$ps/"
                surl = strip(url, '/')
                title = pagevar(surl, :title)
                pubdate = pagevar(surl, :published)
                description = pagevar(surl, :description)
                if isnothing(pubdate)
                    date    = "$ys-$ms-01"
                    days[i] = 1
                else
                    date    = pubdate
                    days[i] = day(date)
                end
                lines[i] = "\n$date: [**$title**]($url)\n"
            end
            # sort by day
            foreach(line -> write(io, line), lines[sortperm(days, rev=true)])
        end
    end
    # markdown conversion adds `<p>` beginning and end but
    # we want to  avoid this to avoid an empty separator
    r = Franklin.fd2html(String(take!(io)), internal=true)
    return r
end

function hfun_published_formatted()
    return Dates.format(locvar(:published), locvar(:date_format))
    return the_formatted_date
end
