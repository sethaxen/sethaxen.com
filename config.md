<!--
Add here global page variables to use throughout your
website.
The website_* must be defined for the RSS to work
-->
@def generate_rss = true
@def website_title = "Seth Axen"
@def website_descr = "Seth Axen"
@def website_url   = "https://sethaxen.com"
@def repo_url = "https://github.com/sethaxen/sethaxen.com"

@def author = "Seth Axen"
@def image = "https://sethaxen.com/assets/seth.jpg"

@def generate_rss = true
@def rss_website_title = "Seth Axen's blog"
@def rss_website_descr = "Random thoughts on math, statistics, and code that I haven't felt like turning into papers."
@def rss_website_url   = "https://sethaxen.com/blog"
@def rss_full_content = true

@def mintoclevel = 2

<!--
Add here files or directories that should be ignored by Franklin, otherwise
these files might be copied and, if markdown, processed by Franklin which
you might not want. Indicate directories by ending the name with a `/`.
-->
@def ignore = ["node_modules/", "franklin", "franklin.pub"]

<!--
Add here global latex commands to use throughout your
pages. It can be math commands but does not need to be.
For instance:
* \newcommand{\phrase}{This is a long phrase to copy.}
-->
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\H}{\mathbb{H}}

<!-- 
Useful commands inspired by the physics package
 -->
\newcommand{\ip}[2]{\left\langle #1, #2 \right\rangle}
\newcommand{\Re}{\operatorname{Re}}
\newcommand{\Im}{\operatorname{Im}}
\newcommand{\sign}{\operatorname{sign}}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
\newcommand{\tr}{\operatorname{tr}}
\newcommand{\dd}{\mathrm{d}}

\newcommand{\details}[1]{
    ~~~<details>~~~
    #1
    ~~~</details>~~~
}
\newcommand{\summary}[1]{
    ~~~
    <summary>~~~
    #1
    ~~~</summary>
    ~~~
}

<!-- syntax \project{name}{image}{url}{tags}{description} -->
\newcommand{\project}[5]{
~~~
<div class="project-card !#4">
<a class="project-image" href="!#3">
<img src="!#2" alt="!#1">
</a>
<a href="!#3">
<h3>!#1</h3>
</a>
<div class="project-description">~~~
!#5
~~~</div>
</div>
~~~
}
