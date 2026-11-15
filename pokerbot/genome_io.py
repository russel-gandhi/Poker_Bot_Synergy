"""Genome file I/O for training only. Not shipped in the submission: the bot never writes files."""
import json

from .agent import genome_from_json, genome_to_json


def save_genome(g, path, meta=None):
    with open(path, "w") as f:
        json.dump({"genome": genome_to_json(g), "meta": meta or {}}, f)


def load_genome(path):
    with open(path) as f:
        return genome_from_json(json.load(f)["genome"])
