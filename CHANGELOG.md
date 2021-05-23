# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.5.0 - 2021-05-23
### Added
* `--core prewarp --query` has been replaced with a hilariously fast version of Hamming on the CPU
    * Hamming distance check is implemented in Cython, with appropriate static types
    * FASTA loader is ignored in favour of reading and processing the target FASTA in a striped fashion
    * This function is not integrated properly into the existing interfaces and needs some proper thought
* `add_seq` on `FastaLoader` manages sequence names and lengths for blocks, and handles sequence conversion
* `TrivialFastaLoader` added to loaders
* `BytesEncoder` added to encoders

### Changed
* `get_length` no longer requires implementing in loader interfaces

### Removed
* `TestFastaLoader` removed from loaders
* `limit` argument removed from `FastaLoader` interface, use `get_block(target_n=limit)` instead
* `--encoder` is no longer a selectable option, as `bytes` is the only encoder that currently works and is automatically selected
* `--loader` is no longer a selectable option, `--loader heng` is automatically used unless `--loader trivial` is required


## 0.3.2 - 2021-05-20
### Changed
* `--core prewarp` is approximately 20x faster than yesterday as the kernel is now precompiled with Cython, but is still a bit rubbish

## 0.3.0 - 2021-05-19
### Added
* `--core` option selects between `warp` (for GPU acceleration) and `prewarp` (for not very fast CPU work)
    * `--core prewarp` is not currently recommended as it is not very good (requires some multiprocessing magic)

### Changed
* Kernel is no longer selected with `--kernel` but selected automatically based on the new `--core` option

## 0.2.1 - 2021-05-13
### Added
* `--query` option specifies input query FASTA to be checked against `--target`

### Changed
* Mandatory `--fasta` option is now named `--target`
* `--hog` option removed and is now selected from the command options
    * Rectangular matrix used when `--query` is specified (previously `--hog some`)
    * Triangular matrix used when `--query` is not specified (previously `--hog all`)
* Moved CHANGELOG to root

### Removed
* `HOGS` no longer defined by `warphog/hogs.py`

## 0.1.2 - 2021-05-11
### Added
* `CHANGELOG.md`
* `test_warphog.py` attempts to check the basic integrity of warphog
