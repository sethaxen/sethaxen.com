<!--
Add here global page variables to use throughout your
website.
The website_* must be defined for the RSS to work
-->
@def generate_rss = true
@def website_title = "Seth Axen"
@def website_descr = "Seth Axen"
@def website_url   = "https://sethaxen.com"
@def repo_url = "https://github.com/sethaxen/sethaxen.github.io"

@def author = "Seth Axen"
@def image = "https://sethaxen.com/assets/seth.jpg"

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
\newcommand{\R}{\mathbb R}
\newcommand{\C}{\mathbb C}

<!-- 
Useful commands inspired by the physics package
 -->
\newcommand{\ip}[2]{\left\langle #1, #2 \right\rangle}
\newcommand{\Re}{\operatorname{Re}}
\newcommand{\Im}{\operatorname{Im}}
\newcommand{\sign}{\operatorname{sign}}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
