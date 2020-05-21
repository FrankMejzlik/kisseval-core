ImageRanker for KISSEval {#mainpage}
======

Library exporting public API for text model evaluation and for user data collection and processing.


## Remarks
- It is NOT thread safe and calling multiple methods in parallel is unsafe
    - Therefore caller must guarantee that no two methods will be called in parallel
    - This will change once this library will be targeted for public use