"""Shim simplejson.

requests tries to replace json with simplejson. Sanic uses requests for running
tests and all non-JSON responses break. In tests this directory is first in
sys.path, thus this file will be imported instead of the real simplejson.
"""
from json import *
