@def title = "Blog"
@def tags = ["blog"]

Random writings that I didn't feel like turning into papers.

~~~
<style>
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  grid-gap: 1rem;
}

.grid > h2, .grid > p {
  background: #f2f2f2;
  padding: 0.5rem 0.5rem 1rem 0.5rem;
  font-size: 0.9em;
  line-height: 1.1rem;
  border-radius: 5px;
  text-align:left;
}

.grid > p a {
  display: block;
  padding-bottom: 0.5em;
}

</style>
<div class="grid">
~~~

{{blogposts}}

~~~
</div>
~~~
