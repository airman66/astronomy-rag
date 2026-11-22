import json
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest import TestCase

from app.catalog.corpus import corpus_digest
from app.catalog.release import verify_release
from app.infra.files import checksum
from app.pipeline.chunks import CHUNK_VERSION
from scripts.catalog_release import publish
from tests.catalog.fixtures import seed_vector_index
from tests.support.catalog import make_term, temporary_dir, write_term


class ReleaseTests(TestCase):
    def setUp(self) -> None:
        self.source = temporary_dir(self).resolve()
        self.target = self.source / "releases"
        write_term(self.source, make_term(1, "Sun", "Солнце", "A star.", "Звезда."))
        seed_vector_index(self.source)
        self.bind_index_to_corpus()

    def bind_index_to_corpus(self):
        index_file = self.source / "index" / "index-manifest.json"
        index = json.loads(index_file.read_text())
        chunk = {
            "chunk_version": CHUNK_VERSION,
            "chunk_count": index["chunk_count"],
            "chunks_sha256": index["chunks_sha256"],
            "corpus_sha256": corpus_digest(self.source),
        }
        (index_file.parent / "chunk-manifest.json").write_text(json.dumps(chunk))
        index.update(corpus_sha256=chunk["corpus_sha256"], chunk_config=chunk)
        index_file.write_text(json.dumps(index))

    def test_changed_corpus_does_not_replace_a_valid_release(self):
        first = publish(self.source, self.target)
        self.assertEqual((self.target / "current").resolve(), first)
        self.assertEqual((first / "models").resolve(), self.source / "models")
        verify_release(first)
        with closing(sqlite3.connect(first / "glossary.db")) as db:
            self.assertEqual(db.execute("SELECT value FROM meta WHERE key='corpus_sha256'").fetchone(),
                             (corpus_digest(self.source),))
        self.assertTrue((first / "processed/terms/1.json").exists())

        write_term(self.source, make_term(1, "Sun", "Солнце", "Our star.", "Наша звезда."))
        with self.assertRaisesRegex(ValueError, "canonical corpus"):
            publish(self.source, self.target)
        self.assertEqual((self.target / "current").resolve(), first)
        verify_release(first)

    def seed_cover(self) -> Path:
        relative_path = Path("media/illustrations/1.webp")
        cover = self.source / relative_path
        cover.parent.mkdir(parents=True)
        cover.write_bytes(b"original cover")
        manifest = self.source / "illustrations" / "manifest.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps({"items": [{
            "term_number": 1, "status": "approved", "path": relative_path.as_posix(), "sha256": checksum(cover),
        }]}))
        return relative_path

    def test_broken_cover_does_not_replace_a_valid_release(self):
        first = publish(self.source, self.target)
        cover = self.source / self.seed_cover()
        cover.write_bytes(b"replaced cover")
        with self.assertRaisesRegex(ValueError, "cover for term 1 has a checksum mismatch"):
            publish(self.source, self.target)
        cover.unlink()
        with self.assertRaisesRegex(ValueError, "cover for term 1 is missing"):
            publish(self.source, self.target)
        self.assertEqual((self.target / "current").resolve(), first)

    def test_verification_rejects_a_changed_or_missing_cover(self):
        relative_path = self.seed_cover()
        release = publish(self.source, self.target)
        verify_release(release)
        cover = release / relative_path
        cover.write_bytes(b"replaced cover")
        with self.assertRaisesRegex(ValueError, "cover for term 1 has a checksum mismatch"):
            verify_release(release)
        cover.unlink()
        with self.assertRaisesRegex(ValueError, "cover for term 1 is missing"):
            verify_release(release)

    def test_review_pdfs_are_required_at_publication_and_verified_afterwards(self):
        first = publish(self.source, self.target)
        original = Path("raw/reviews/review.pdf")
        translation = Path("media/ru/review-pdf/oae-document-review.pdf")
        files = {original: b"%PDF original", translation: b"%PDF translation"}
        article = make_term(1, "Review", "Обзор", "A review.", "Обзор.")
        article["id"] = "article:oae-document-review"
        article["source"].update(
            type="oae_review_document", pdf_path=original.as_posix(),
        )
        for path, content in files.items():
            (self.source / path).parent.mkdir(parents=True)
            (self.source / path).write_bytes(content)
        article["source"]["sha256"] = checksum(self.source / original)
        canonical = self.source / "processed/articles/oae-document-review.json"
        canonical.parent.mkdir()
        canonical.write_text(json.dumps(article))
        self.bind_index_to_corpus()

        current = self.target / "current"
        for path, locale in ((original, "en"), (translation, "ru")):
            with self.subTest(locale=locale):
                (self.source / path).unlink()
                with self.assertRaisesRegex(ValueError, f"PDF {locale} .* is missing"):
                    publish(self.source, self.target)
                self.assertEqual(current.resolve(), first)
                (self.source / path).write_bytes(files[path])
        (self.source / original).write_bytes(b"%PDF damaged")
        with self.assertRaisesRegex(ValueError, "PDF en .* checksum mismatch"):
            publish(self.source, self.target)
        self.assertEqual(current.resolve(), first)
        (self.source / original).write_bytes(files[original])

        release = publish(self.source, self.target)
        verify_release(release)
        (release / translation).write_bytes(b"%PDF changed translation")
        with self.assertRaisesRegex(ValueError, "PDF checksums do not match"):
            verify_release(release)
        (release / original).unlink()
        with self.assertRaisesRegex(ValueError, "PDF en .* is missing"):
            verify_release(release)

    def test_damaged_catalogue_media_fails_verification_and_republication(self):
        first = publish(self.source, self.target)
        term = make_term(1, "Sun", "Солнце", "A star.", "Звезда.")
        media = {"id": "media:sun", "kind": "media", "review": {"status": "draft"}}
        for filename, path_key, digest_key in (
            ("original/sun.png", "original_path", "sha256"),
            ("ru/sun.png", "ru_path", "ru_sha256"),
        ):
            path = self.source / "media" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(filename.encode())
            media[path_key] = path.relative_to(self.source).as_posix()
            media[digest_key] = checksum(path)
        term["media"] = [media]
        write_term(self.source, term)
        self.bind_index_to_corpus()
        release = publish(self.source, self.target)
        verify_release(release)
        current = self.target / "current"
        current.unlink()
        current.symlink_to(first.name)

        original = release / "media/original/sun.png"
        original.write_bytes(b"changed image")
        with self.assertRaisesRegex(ValueError, "checksum: media/original/sun.png"):
            verify_release(release)
        original.write_bytes(b"original/sun.png")
        (release / "media/ru/sun.png").unlink()
        with self.assertRaisesRegex(ValueError, "missing: media/ru/sun.png"):
            verify_release(release)
        with self.assertRaisesRegex(ValueError, "missing: media/ru/sun.png"):
            publish(self.source, self.target)
        self.assertEqual(current.resolve(), first)
