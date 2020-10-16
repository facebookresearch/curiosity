#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

import logging


def get_logger(name):
    log = logging.getLogger(name)

    if len(log.handlers) < 2:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        fh = logging.FileHandler("curio-model.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        log.addHandler(fh)
        log.addHandler(sh)
        log.setLevel(logging.INFO)
    return log
