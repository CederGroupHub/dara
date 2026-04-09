import tempfile
import unittest
import warnings
from pathlib import Path

from dara.cif2str import cif2str
from dara.utils import read_phase_name_from_str


class TestCif2Str(unittest.TestCase):
    """Test the cif2str function."""

    def setUp(self):
        """Set up the test."""
        self.cif_paths = list((Path(__file__).parent / "test_data").glob("*.cif"))

    def test_cif2str(self):
        """Test the cif2str function with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for cif_path in self.cif_paths:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    str_path = cif2str(cif_path, "", tmpdir)
                    self.assertTrue(len(w) == 0)

                self.assertTrue(str_path.exists())
                self.assertTrue(str_path.is_file())
                self.assertTrue(str_path.suffix == ".str")
                self.assertTrue(str_path.stem == cif_path.stem)

                ref_str_path = cif_path.parent / (cif_path.stem + ".str")

                ref_phase_name = read_phase_name_from_str(ref_str_path)
                phase_name = read_phase_name_from_str(str_path)
                self.assertTrue(ref_phase_name == phase_name)

    def test_cif2str_custom_params(self):
        """Test the cif2str function with custom_params and custom_params_map."""
        if not self.cif_paths:
            self.skipTest("No CIF files found for testing.")

        # Just use the first available CIF for this test
        cif_path = self.cif_paths[0]
        
        custom_params = ["PARAM=Bglobal=0.05_0.01^0.20 //", "PARAM=BO=0.1_0.02^0.3 //"]
        custom_params_map = {
            "*": {"TDS": "Bglobal"},
            "O": {"TDS": "BO", "Occ": "OccO"}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            str_path = cif2str(
                cif_path, 
                working_dir=tmpdir, 
                custom_params=custom_params, 
                custom_params_map=custom_params_map
            )

            self.assertTrue(str_path.exists())

            with open(str_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Verify that custom lines were successfully injected
            for line in custom_params:
                self.assertIn(line, content)

            # Verify the wildcard fallback applied Bglobal
            self.assertIn("TDS=Bglobal", content)

            # Verify that the hardcoded room-temp TDS was completely overwritten by the wildcard
            self.assertNotIn("TDS=0.010000", content)

            # If the test CIF happens to contain oxygen, verify the specific element overrides took precedence
            if "E=O" in content or "O-" in content or "O+" in content:
                self.assertIn("TDS=BO", content)
                self.assertIn("Occ=OccO", content)