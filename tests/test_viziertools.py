"""Tests for the Vizier query guard (tools/viziertools.py): the
empty/failed-response handling added with the Gaia DR3 switch (e20fcf1d),
exercised with an injected fake querier - no network involved."""

import unittest

from astropy.coordinates import SkyCoord
from astropy.table import Table

from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.viziertools import _query_vizier_first_table

_COORD = SkyCoord(ra=10.0, dec=-5.0, unit="deg")


class FakeVizier:
    """Stands in for an astroquery Vizier instance."""

    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = []

    def query_region(self, skycoord, radius=None, catalog=None):
        self.calls.append({"radius": radius, "catalog": catalog})
        if self.exc is not None:
            raise self.exc
        return self.result


class TestQueryVizierFirstTable(unittest.TestCase):
    def test_returns_first_table(self):
        table = Table({"ra": [1.0]})
        fake = FakeVizier(result=[table])
        out = _query_vizier_first_table(
            fake, _COORD, radius=0.5, catalog="I/355/gaiadr3", name="Gaia DR3"
        )
        self.assertIs(out, table)
        self.assertEqual(fake.calls[0]["catalog"], "I/355/gaiadr3")

    def test_empty_result_raises_pipeline_error(self):
        for result in ([], None):
            fake = FakeVizier(result=result)
            with self.assertRaises(PipelineValueError) as ctx:
                _query_vizier_first_table(
                    fake, _COORD, radius=0.5, catalog="cat", name="Gaia DR3"
                )
            self.assertIn("returned no table", str(ctx.exception))

    def test_query_exception_raises_pipeline_error(self):
        fake = FakeVizier(exc=TimeoutError("CDS down"))
        with self.assertRaises(PipelineValueError) as ctx:
            _query_vizier_first_table(
                fake, _COORD, radius=0.5, catalog="cat", name="Gaia DR3"
            )
        self.assertIn("query failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
