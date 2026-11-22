import json
import sqlite3
from contextlib import closing
from unittest import TestCase

from app.catalog.corpus import corpus_digest
from app.catalog.release import verify_release
from app.pipeline.chunks import CHUNK_VERSION
from scripts.catalog_release import publish
from tests.catalog.fixtures import seed_vector_index
from tests.support.catalog import make_term, temporary_dir, write_term


class ReleaseTests(TestCase):
    def test_changed_corpus_does_not_replace_a_valid_release(self):
        source = temporary_dir(self).resolve()
        write_term(source, make_term(1, "Sun", "Солнце", "A star.", "Звезда."))
        seed_vector_index(source)
        index_file = source / "index" / "index-manifest.json"
        index = json.loads(index_file.read_text())
        chunk = {
            "chunk_version": CHUNK_VERSION,
            "chunk_count": index["chunk_count"],
            "chunks_sha256": index["chunks_sha256"],
            "corpus_sha256": corpus_digest(source),
        }
        (index_file.parent / "chunk-manifest.json").write_text(json.dumps(chunk))
        index.update(corpus_sha256=chunk["corpus_sha256"], chunk_config=chunk)
        index_file.write_text(json.dumps(index))

        target = source / "releases"
        first = publish(source, target)
        self.assertEqual((target / "current").resolve(), first)
        self.assertEqual((first / "models").resolve(), source / "models")
        verify_release(first)
        with closing(sqlite3.connect(first / "glossary.db")) as db:
            self.assertEqual(db.execute("SELECT value FROM meta WHERE key='corpus_sha256'").fetchone(),
                             (chunk["corpus_sha256"],))
        self.assertTrue((first / "processed/terms/1.json").exists())

        write_term(source, make_term(1, "Sun", "Солнце", "Our star.", "Наша звезда."))
        with self.assertRaisesRegex(ValueError, "canonical corpus"):
            publish(source, target)
        self.assertEqual((target / "current").resolve(), first)
        verify_release(first)
