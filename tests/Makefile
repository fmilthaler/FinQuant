default: test

test:rempltshow
	@pytest -v .

rempltshow:plt_agg_backend
	@-sed -i 's/plt.show()//g' test_Example*.py

plt_agg_backend:
	@-sed -i '/^import matplotlib\.pyplot as plt$$/a \plt.switch_backend("Agg")' test_Example*.py

clean:
	@-rm -rf *.pyc test_Example*.py __pycache__

