'''
-------
License
-------

It is released under the MIT license.

    Copyright (c) 2019 Robert Grigoroiu

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

try:
    from setuptools import setup, Extension
except Exception:
    from distutils import setup, Extension
from Cython.Build import cythonize

README = open("./README.md").read()

metadata = dict(
    name='bulknlp',
    version='2.0.0',
    description='Efficient NLP algorithms',
    long_description=README,
    long_description_content_type="text/markdown",
    author='BulkNLP',
    author_email='bulknlp@mail.com',
    url='https://github.com/bulknlp/bulknlp',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ]
)

setup(
    ext_modules=[Extension('fastDamerauLevenshtein', ['fastDamerauLevenshtein/fastDamerauLevenshtein.c'])],
    **metadata
)

setup(
    ext_modules = cythonize("bulknlp/DamerauLevenshtein.pyx")
)
