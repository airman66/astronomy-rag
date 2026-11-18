from unittest import TestCase
from unittest.mock import patch

from app.pipeline import importer
from tests.support.catalog import temporary_dir


class ImporterCase(TestCase):
    def setUp(self):
        self.root = temporary_dir(self) / "data"
        self.enterContext(patch.multiple(importer, DATA=self.root, REFRESH=False))
