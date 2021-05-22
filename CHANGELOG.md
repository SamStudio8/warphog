# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.5.0 - Unreleased
### Added
* `--core prewarp-beta` is a hilariously fast version of Hamming on the CPU, after adding appropriate static types to Cython definition and re-writing the FASTA parser to read in a striped fashion
    * Completely unintegrated into the existing codebase which previously loaded sequences up-front (for the GPU) so needs some thought
* `add_seq` on `FastaLoader` manages sequence names and lengths for blocks, and handles sequence conversion
* `TrivialFastaLoader` added to loaders
* `BytesEncoder` added to encoders

### Changed
* `get_length` no longer requires implementing in loader interfaces

### Removed
* `TestFastaLoader` removed from loaders
* `limit` argument removed from `FastaLoader` interface, use `get_block(target_n=limit)` instead


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
