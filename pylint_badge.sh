#!/bin/bash
pip install anybadge
pylint $* -f text --exit-zero > .pylintout 2> .pylinterror
mkdir -p public/pylint/
anybadge -l pylint -v $(cat .pylintout | tail -n2 | awk '{print $7}' | cut -d"/" -f1) -f ./public/pylint/badge.svg 5=red 8=orange 9=yellow 9.5=yellowgreen 10=green
echo "<pre>" > ./public/pylint/index.html
cat .pylinterror >> ./public/pylint/index.html
cat .pylintout >> ./public/pylint/index.html
echo "</pre>" >> ./public/pylint/index.html
cat .pylinterror
cat .pylintout
