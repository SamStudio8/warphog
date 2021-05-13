# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
