default: test

test:rempltshow
	@pytest -v .

rempltshow:
	@-sed -i 's/plt.show()//g' test_Example*.py

clean:
	@-rm -rf *.pyc test_Example*.py __pycache__

