ImageRanker for KISSEval {#mainpage}
======

Library exporting public API for text model evaluation and for user data collection and processing. Most of the documentation is actually in the code itself.


# Remarks
- It is NOT thread safe and calling multiple methods in parallel is unsafe
    - Therefore caller must guarantee that no two methods will be called in parallel
    - This will change once this library will be targeted for public use

# Overview
This image shows how this library is structured.
@image html kisseval_overview.png "System overview"

# License
Copyright 2020 Frantisek Mejzlik <frankmejzlik@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.