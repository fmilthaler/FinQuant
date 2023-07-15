default: test

test:rempltshow
	@pytest -v .

rempltshow:plt_agg_backend
	@-sed -i 's/plt.show()//g' test_Example*.py

plt_agg_backend:set_api_key_example
	@-sed -i '/^import matplotlib\.pyplot as plt$$/a \plt.switch_backend("Agg")' test_Example*.py

set_api_key_example:
	@-sed -i '/^from finquant\.portfolio import build_portfolio$$/a \import os;import quandl;quandl\.ApiConfig\.api_key = os\.getenv("QUANDLAPIKEY")' test_Example-Build-Portfolio-from-web.py

clean:
	@-rm -rf *.pyc test_Example*.py __pycache__

