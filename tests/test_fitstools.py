import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from vircampype.tools.fitstools import (
    add_float_to_header,
    add_int_to_header,
    add_key_primary_hdu,
    add_str_to_header,
    check_card_value,
    delete_keyword_from_header,
    make_card,
    make_cards,
    mjd2dateobs,
    read_fits_headers,
)


class TestCheckCardValue(unittest.TestCase):
    def test_string(self):
        self.assertEqual(check_card_value("hello"), "hello")

    def test_int(self):
        self.assertEqual(check_card_value(42), 42)

    def test_float(self):
        self.assertAlmostEqual(check_card_value(3.14), 3.14)

    def test_numpy_float(self):
        val = check_card_value(np.float64(2.71))
        self.assertIsInstance(val, (float, np.floating))

    def test_numpy_int(self):
        val = check_card_value(np.int32(10))
        self.assertIsInstance(val, (int, np.integer))

    def test_callable(self):
        val = check_card_value(np.nanmedian)
        self.assertEqual(val, "median")

    def test_other_type_cast_to_str(self):
        val = check_card_value([1, 2, 3])
        self.assertIsInstance(val, str)


class TestMakeCard(unittest.TestCase):
    def test_basic_card(self):
        card = make_card("TESTKEY", 42, comment="A test")
        self.assertIsInstance(card, fits.Card)
        self.assertEqual(card.keyword, "TESTKEY")
        self.assertEqual(card.value, 42)

    def test_uppercase(self):
        card = make_card("lowercase", "value")
        self.assertEqual(card.keyword, "LOWERCASE")

    def test_no_uppercase(self):
        card = make_card("MixedCase", "value", upper=False)
        self.assertEqual(card.keyword, "MixedCase")

    def test_too_long_raises(self):
        with self.assertRaises(ValueError):
            make_card("KEY", "x" * 80, comment="comment")

    def test_whitespace_collapse(self):
        card = make_card("HIERARCH  ESO  KEY", "value")
        self.assertNotIn("  ", card.keyword)


class TestMakeCards(unittest.TestCase):
    def test_basic(self):
        cards = make_cards(["KEY1", "KEY2"], [1, 2])
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0].value, 1)
        self.assertEqual(cards[1].value, 2)

    def test_with_comments(self):
        cards = make_cards(["KEY1", "KEY2"], [1, 2], comments=["c1", "c2"])
        self.assertEqual(len(cards), 2)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            make_cards(["KEY1"], [1, 2])

    def test_mismatched_comments(self):
        with self.assertRaises(ValueError):
            make_cards(["KEY1", "KEY2"], [1, 2], comments=["c1"])

    def test_not_list_raises(self):
        with self.assertRaises(TypeError):
            make_cards("KEY1", [1])


class TestDeleteKeywordFromHeader(unittest.TestCase):
    def test_existing_keyword(self):
        hdr = fits.Header()
        hdr["TESTKEY"] = 42
        result = delete_keyword_from_header(hdr, "TESTKEY")
        self.assertNotIn("TESTKEY", result)

    def test_missing_keyword(self):
        hdr = fits.Header()
        result = delete_keyword_from_header(hdr, "NONEXISTENT")
        self.assertNotIn("NONEXISTENT", result)


class TestAddToHeader(unittest.TestCase):
    def test_add_float(self):
        hdr = fits.Header()
        add_float_to_header(hdr, "TESTF", 3.14159, decimals=3, comment="pi")
        self.assertIn("TESTF", hdr)
        self.assertAlmostEqual(float(hdr["TESTF"]), 3.142, places=3)

    def test_add_float_various_decimals(self):
        for dec in range(1, 7):
            hdr = fits.Header()
            add_float_to_header(hdr, "TEST", 1.123456, decimals=dec)
            self.assertIn("TEST", hdr)

    def test_add_float_invalid_decimals(self):
        hdr = fits.Header()
        with self.assertRaises(ValueError):
            add_float_to_header(hdr, "TEST", 1.0, decimals=7)

    def test_add_int(self):
        hdr = fits.Header()
        add_int_to_header(hdr, "TESTI", 42, comment="answer")
        self.assertIn("TESTI", hdr)
        self.assertEqual(hdr["TESTI"], 42)

    def test_add_str(self):
        hdr = fits.Header()
        add_str_to_header(hdr, "TESTS", "hello", comment="greeting")
        self.assertIn("TESTS", hdr)
        self.assertEqual(hdr["TESTS"], "hello")

    def test_add_str_no_comment(self):
        hdr = fits.Header()
        add_str_to_header(hdr, "TESTS", "world")
        self.assertEqual(hdr["TESTS"], "world")

    def test_remove_before(self):
        hdr = fits.Header()
        hdr["MYKEY"] = 1
        add_int_to_header(hdr, "MYKEY", 2, remove_before=True)
        self.assertEqual(hdr["MYKEY"], 2)

    def test_no_remove_before(self):
        hdr = fits.Header()
        hdr["MYKEY"] = 1
        add_int_to_header(hdr, "MYKEY", 2, remove_before=False)
        # Both entries should exist (duplicate keys)
        values = [hdr[i] for i in range(len(hdr)) if hdr.cards[i].keyword == "MYKEY"]
        self.assertIn(1, values)
        self.assertIn(2, values)


class TestMJD2DateObs(unittest.TestCase):
    def test_known_date(self):
        # MJD 51544.5 is 2000-01-01T12:00:00
        result = mjd2dateobs(51544.5)
        self.assertIn("2000-01-01", result)

    def test_type(self):
        result = mjd2dateobs(59000.0)
        self.assertIsInstance(result, str)


class TestReadFitsHeaders(unittest.TestCase):
    def test_read_simple(self):
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = f.name
        try:
            hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
            hdu.header["TESTKEY"] = 42
            hdu.writeto(path, overwrite=True)
            headers = read_fits_headers(path)
            self.assertIsInstance(headers, list)
            self.assertEqual(len(headers), 1)
            self.assertEqual(headers[0]["TESTKEY"], 42)
        finally:
            os.remove(path)

    def test_multi_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = f.name
        try:
            primary = fits.PrimaryHDU()
            ext1 = fits.ImageHDU(data=np.zeros((5, 5)))
            ext2 = fits.ImageHDU(data=np.zeros((5, 5)))
            hdulist = fits.HDUList([primary, ext1, ext2])
            hdulist.writeto(path, overwrite=True)
            headers = read_fits_headers(path)
            self.assertEqual(len(headers), 3)
        finally:
            os.remove(path)


class TestAddKeyPrimaryHDU(unittest.TestCase):
    def test_add_key(self):
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = f.name
        try:
            hdu = fits.PrimaryHDU(data=np.zeros((5, 5)))
            hdu.writeto(path, overwrite=True)
            add_key_primary_hdu(path, key="MYKEY", value=99, comment="test")
            with fits.open(path) as hdul:
                self.assertEqual(hdul[0].header["MYKEY"], 99)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
