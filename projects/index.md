@def title = "Selected Projects"
@def tags = ["projects"]

~~~
<script src="https://unpkg.com/masonry-layout@4/dist/masonry.pkgd.min.js"></script>
<script src="https://unpkg.com/imagesloaded@5/imagesloaded.pkgd.min.js"></script>
<script>
    document.addEventListener('DOMContentLoaded', function () {
        // If image not provided, don't show it
        var imageList = document.getElementsByTagName('img');
        for (var i = 0; i < imageList.length; i++) {
            if (imageList[i].getAttribute('src').trim() === '') {
                var parentDiv = imageList[i].closest('.project-image'); // Find the closest parent div with the class 'project-image'
                if (parentDiv) {
                    parentDiv.parentNode.removeChild(parentDiv); // Remove the parent div from the DOM
                    i--; // Decrement because we just decreased size of list
                }
            }
        }
        
        var gridList = document.querySelectorAll(".project-list");
        gridList.forEach(grid => {
            var msnry = new Masonry(grid, {
                itemSelector: '.project-card',
                columnWidth: 250,
                gutter: 15
            });

            // Setup imagesLoaded only once per grid
            imagesLoaded(grid).on('progress', function() {
                msnry.layout();
            });

            // Setup buttons and filtering
            document.querySelectorAll('.filter-btn').forEach(button => {
                button.addEventListener('click', function() {
                    const filter = this.getAttribute('data-filter');
                    const projectCards = grid.querySelectorAll('.project-card');

                    projectCards.forEach(card => {
                        if (filter === 'all' || card.classList.contains(filter)) {
                            card.style.display = ''; // Ensure the card is visible
                        } else {
                            card.style.display = 'none'; // Hide the card
                        }
                    });

                    // Remove active class from all buttons
                    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
                    
                    // Add active class to the clicked button
                    this.classList.add('active');

                    // Relayout Masonry after filtering
                    msnry.layout();
                });
            });
        });
    })
</script>
~~~

# Selected Projects

~~~
<div id="toolbar" class="filter-toolbar">
    <button class="filter-btn" data-filter="all">All</button>
    <button class="filter-btn" data-filter="machine-learning">Machine Learning</button>
    <button class="filter-btn" data-filter="python">Python</button>
    <button class="filter-btn" data-filter="julia">Julia</button>
    <button class="filter-btn" data-filter="visualization">Visualization</button>
</div>
~~~

~~~<div class='project-list'>~~~

\project{ArviZ: exploratory analysis of Bayesian models}{https://julia.arviz.org/ArviZ/stable/assets/logo.png}{https://www.arviz.org/}{machine-learning visualization julia python}{
Core contributor, governance member, and primary dev of Julia packages.
}

\project{PollenClim}{https://images.spr.so/cdn-cgi/imagedelivery/j42No7y-dcokJuNgXeA0ig/e29fd6b9-d490-46c2-b897-5c5d91ab1cd4/pollenclim_what-2-1/w=1080,quality=80,fit=scale-down}{https://mlcolab.org/resources/integration-of-paleoclimate-models-and-proxies}{machine-learning python sciml}{
Building a consensus probabilistic model of paleoclimate from simulations of climate models and fossilized pollen data.
}

\project{Pathfinder.jl}{https://images.spr.so/cdn-cgi/imagedelivery/j42No7y-dcokJuNgXeA0ig/2514118c-8d6f-4aa3-b768-b1d029f7eea8/index/w=1200,quality=80,fit=scale-down}{https://mlcolab.org/resources/pathfinderjl-early-diagnostics-for-probabilistic-models-and-faster-mcmc-warmup}{machine-learning julia}{
Accelerating Bayesian inference and early diagnostics for probabilistic models.
}

\project{Transforms}{}{https://github.com/mjhajharia/transforms}{machine-learning}{
Working out and benchmarking efficient invertible transforms from unconstrained to constrained spaces for probabilistic programming.
See Stan implementations for [probability vectors](https://github.com/mjhajharia/transforms) and for [orthogonal matrices](https://github.com/sethaxen/stan_semiorthogonal_transforms).
Papers in prep.
}

\project{e3fp}{/assets/project_images/e3fp.jpeg}{https://github.com/keiserlab/e3fp}{machine-learning python}{
3D-aware fingerprint representations of small molecules for machine learning.
}

\project{JuliaManifolds}{https://juliamanifolds.github.io/juliamanifolds/assets/logo.png}{https://juliamanifolds.github.io/}{julia}{
Co-author of [ManifoldsBase.jl](https://github.com/JuliaManifolds/ManifoldsBase.jl) interface for defining manifolds and algorithms on those manifolds, along with implementations in [Manifolds.jl](https://github.com/JuliaManifolds/Manifolds.jl). [paper](https://doi.org/10.1145/3618296)
}

\project{ChainRules}{https://juliadiff.org/ChainRulesCore.jl/stable/assets/logo.svg}{https://juliadiff.org/ChainRulesCore.jl/stable/}{julia}{
Core contributor to tools for defining and testing AD rules across the Julia ecosystem.
}

\project{Pathfinder Benchmarks}{https://images.spr.so/cdn-cgi/imagedelivery/j42No7y-dcokJuNgXeA0ig/3ecbe140-32e4-4302-a02b-b76d66655312/Pathfinder_benchmarks_poster_BayesComp_2023-1/w=1000}{https://mlcolab.org/public-events/faster-bayesian-inference-with-pathfinder}{machine-learning julia}{
Experimenting with different uses of Pathfinder for accelerating Bayesian modeling workflows.
}

<!-- \project{Frequency-informed linear discriminative learning}{}{}{machine-learning}{
} -->

\project{Calibrated confidence bands for ECDFs}{/assets/project_images/ecdf_confidence_bands.png}{https://nbviewer.org/gist/sethaxen/06c83cace937a19dd55d4b6ccedbec82}{machine-learning visualization python}{
Numerical experiments testing the calibration of confidence bands for ECDF plots.
}

\project{Intro to Supervised learning}{https://user-images.githubusercontent.com/8673634/161941600-b1c31af3-df9b-4481-bb6f-1e25a4f849d2.gif}{https://mlcolab.github.io/IntroML.jl/2023.07.14/supervised_learning.html}{machine-learning visualization}{
Interactive teaching resource on supervised learning, from linear regression to neural networks.
Developed for the [IntroML workshop series](https://mlcolab.org/resources/introml-november-2023-workshop-materials) at the University of Tübingen.
}

\project{ManifoldMeasures}{}{https://github.com/JuliaManifolds/ManifoldMeasures.jl}{machine-learning julia}{
Lightweight implementations of common manifold distributions for Bayesian inference.
}

~~~</div>~~~
