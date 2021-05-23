# warphog
travel hamming distances at warp speed

## About

Warphog is a Python tool that can rapidly calculate edit distances between strings of equal length using either the CPU (`--core prewarp`), or a CUDA-enabled GPU (`--core warp`).
Edit distance is calculated simply as the Hamming distance between the two sequences, but additionally allows users to configure an `alphabet` that defines equivalent character pairs.
Warphog can compare a set of `--query` sequences against a set of `--target` sequences, or compare all `--target` sequences against each-other. The distance between every compared pair is returned.

Although initially designed for high-throughput distance calculations on a GPU, small numbers of `--query` sequences can be compared against a large number of `--target` sequences on the CPU (using `--core prewarp`) in a surprisingly short amount of time.
CPU performance is mostly gained by pre-compiling the function that computes the Hamming distance with Cython. Additionally, `--query` offers the option to instantiate multiple file handlers on the `--target` source and avoids sequences moving around in RAM by processing distances as the `--target` is being read.
CUDA aside, GPU performance is primarily gained by pre-compiling the CUDA kernel before execution with `__device__` variables containing the chosen alphabet lookup matrix for quick(ish) character equivalence checking at runtime.

Warphog was written as an experimental exercise in CUDA and Cython. While not formally supported right now, issues and PRs are welcome if this is of use to somebody.

### Performance guide

Your mileage will vary but here is how Warphog runs on a 24-core server and a delightfully fancy GV100 according to some scrawls in my notebook:

| Query Size | Target Size | Sequence Sizes | Time (`--core warp`)| Time (`--core prewarp -t24`) | Time (`--core prewarp -t12`) | Time (`--core prewarp -t1`) |
|------------|-------------|----------------|---------------------|------------------------------|------------------------------|-----------------------------|
| 1          | 557,927     | 29,903         | 1:10* | 0:03 | 0:06 | 0:58 |
| 10         | "           | "              | 1:10* | 0:22 | 0:35 | 5:17 |
| 100        | "           | "              | 1:12  | 3:17 | 5:28 | -    |
| 1,000      | "           | "              | 3:30  | 29:12| -    | -    |
| -          | 10          | "              | 0:01* | x    | x    | 0:01 |
| -          | 100         | "              | 0:01* | x    | x    | 0:01 |
| -          | 1,000       | "              | 0:02* | x    | x    | 0:22 |
| -          | 10,000      | "              | 0:16  | x    | x    | 36:19|

Note for small numbers of `--query` sequences, the GPU is inefficient as the loading of sequences to RAM to be dumped onto the GPU has not been optimised.
Indeed in the cases marked (*), the actual on-GPU time was one second or less. Cases marked (-) just means I couldn't be bothered to run the test, (x) denotes cases where multiprocessing is not supported when using `--core prewarp` without `--query`.

### Limitations

* Warphog double handles input data in order to load it into memory, then onto the GPU, this could likely be optimised away
* Warphog does not make any checks as to whether the choice of `--query` and `--target` will end up using all your host or device RAM (and may well attempt to do so)
* Static types are fixed to widths that worked for our use case but larger sequences or datasets may overflow the widths used without checking
* Some implementations of base classes are currently overriden in a way that breaks their interchangeability
* Warphog only concerns itself with the distances themselves and does not return the location of the edits (although this could be implemented quite easily)
* Test suite could be improved

## Housekeeping
### Install

    python setup.py install

### Test

    python setup.py build_ext --inplace # build the Cython interfaces in place (possibly not ideal)
    pytest

## Usage

### Run a query against a larger dataset on the CPU

    warphog --query query.fa --target targets.fa --core prewarp -o result.txt

### Run a query against a larger dataset on the GPU

    warphog --query query.fa --target targets.fa --core warp -o out.txt

### Run targets against eachother on the GPU, printing results for pairs with an edit distance of 5 or less

    warphog --target targets.fa --core warp -o out.txt -k 5