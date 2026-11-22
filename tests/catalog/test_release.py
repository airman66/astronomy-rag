import json
import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest import TestCase

from app.catalog.corpus import corpus_digest
from app.catalog.index import load_index
from app.catalog.media import local_file
from app.catalog.release import RELEASE_MANIFEST, release_manifest, verify_release
from app.infra.files import checksum
from scripts.catalog_release import publish
from tests.catalog.fixtures import bind_vector_index, seed_vector_index
from tests.support.catalog import make_term, temporary_dir, write_term


class ReleaseTests(TestCase):
    def setUp(self) -> None:
        self.source = temporary_dir(self).resolve()
        self.target = self.source / "releases"
        write_term(self.source, make_term(1, "Sun", "Солнце", "A star.", "Звезда."))
        seed_vector_index(self.source)
        self.bind_index_to_corpus()

    def bind_index_to_corpus(self):
        bind_vector_index(self.source, corpus_digest(self.source))

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

    def test_publication_reuses_version_after_only_source_mtime_changes(self):
        first = publish(self.source, self.target)
        record = self.source / "processed/terms/1.json"
        stat = record.stat()
        os.utime(record, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

        second = publish(self.source, self.target)
        self.assertEqual(second, first)
        self.assertEqual((self.target / "current").resolve(), first)
        verify_release(second)

    def test_republication_rejects_another_valid_release_under_the_same_id(self):
        first = publish(self.source, self.target)
        term = self.source / "processed/terms/1.json"
        original = term.read_bytes()
        write_term(self.source, make_term(1, "Sun", "Солнце", "Our star.", "Наша звезда."))
        self.bind_index_to_corpus()
        second = publish(self.source, self.target)
        first.rename(self.target / "backup")
        shutil.copytree(second, first, symlinks=True)

        with self.assertRaisesRegex(ValueError, "does not match its release ID"):
            verify_release(first)
        term.write_bytes(original)
        self.bind_index_to_corpus()
        with self.assertRaisesRegex(ValueError, "does not match its release ID"):
            publish(self.source, self.target)
        self.assertEqual((self.target / "current").resolve(), second)

    def test_publication_preserves_legacy_media_from_another_data_root(self):
        term = make_term(1, "Sun", "Солнце", "A star.", "Звезда.")
        media = {"id": "media:sun", "kind": "media", "review": {"status": "draft"}}
        for locale_dir, field, digest_key in (
            ("original", "original_path", "sha256"), ("ru", "ru_path", "ru_sha256"),
        ):
            path = self.source / "media" / locale_dir / "sun.png"
            path.parent.mkdir(parents=True)
            path.write_bytes(locale_dir.encode())
            media[field] = str(path)
            media[digest_key] = checksum(path)
        term["media"] = [media]
        write_term(self.source, term)
        self.bind_index_to_corpus()

        release = publish(self.source, self.target)
        (self.source / "media").rename(self.source / "moved-media")
        verify_release(release)
        self.assertEqual(corpus_digest(release), corpus_digest(self.source))
        self.assertEqual(local_file(release, media["original_path"]), release / "media/original/sun.png")
        self.assertEqual(local_file(release, media["ru_path"]), release / "media/ru/sun.png")

    def test_changed_model_location_gets_a_separate_release(self):
        first = publish(self.source, self.target)
        second_source = temporary_dir(self).resolve()
        shutil.copytree(self.source, second_source, dirs_exist_ok=True, ignore=shutil.ignore_patterns("releases"))
        (self.source / "models" / "small.gguf").unlink()

        second = publish(second_source, self.target)
        self.assertNotEqual(first, second)
        self.assertEqual((first / "models").resolve(), self.source / "models")
        self.assertEqual((second / "models").resolve(), second_source / "models")
        self.assertEqual((self.target / "current").resolve(), second)
        load_index(second / "index", second / "models")
        (second / "models").unlink()
        (second / "models").symlink_to("../../models")
        with self.assertRaisesRegex(ValueError, "does not match its manifest"):
            verify_release(second)

    def test_schema_two_releases_remain_readable(self):
        release = publish(self.source, self.target)
        manifest = release_manifest(release, schema_version=2)
        path = release / RELEASE_MANIFEST
        path.write_text(json.dumps(manifest))
        previous = release.rename(self.target / checksum(path))
        verify_release(previous)

    def test_contradictory_chunk_provenance_preserves_current(self):
        first = publish(self.source, self.target)
        write_term(self.source, make_term(1, "Sun", "Солнце", "Changed.", "Изменено."))
        path = self.source / "index" / "index-manifest.json"
        manifest = json.loads(path.read_text())
        manifest["corpus_sha256"] = corpus_digest(self.source)
        path.write_text(json.dumps(manifest))

        with self.assertRaisesRegex(ValueError, "Chunk manifest does not match index manifest"):
            publish(self.source, self.target)
        self.assertEqual((self.target / "current").resolve(), first)

    def test_release_verifies_corpus_inside_sqlite_even_with_updated_file_hash(self):
        release = publish(self.source, self.target)
        database = release / "glossary.db"
        with closing(sqlite3.connect(database)) as db:
            db.execute("UPDATE meta SET value = 'different corpus' WHERE key = 'corpus_sha256'")
            db.commit()
        path = release / RELEASE_MANIFEST
        manifest = json.loads(path.read_text())
        manifest["catalog_sha256"] = checksum(database)
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "must match the canonical corpus"):
            verify_release(release)

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
