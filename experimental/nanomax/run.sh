#!/usr/bin/bash
for k in {134..423}; do python recall.py 4 $k; python recall.py 2 $k; done
