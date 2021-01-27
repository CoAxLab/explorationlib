.PHONY: book
book: 
	-rm -rf book/_build/
	jupyter-book build book
