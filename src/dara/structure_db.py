"""Interact with the (local) ICSD database."""

from __future__ import annotations

import itertools
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core import Composition
from tqdm.contrib.concurrent import thread_map

from dara.cif import Cif
from dara.data import COMMON_GASES
from dara.settings import DaraSettings
from dara.utils import copy_and_rename_files, get_logger

logger = get_logger(__name__)

DARA_SETTINGS = DaraSettings()

PATH_TO_COD = DARA_SETTINGS.PATH_TO_COD
PATH_TO_ICSD = DARA_SETTINGS.PATH_TO_ICSD


class StructureDatabase(MSONable, metaclass=ABCMeta):
    """Base class to interact with CIF files from an experimental database of
    structures. This class is subclassed by ICSDDatabase, CODDatabase, etc.
    """

    def __init__(self, path_to_cifs: Path | str | None = None):
        """
        Initialize an object for interacting with CIFs from a structure database.

        Args:
            path_to_cifs: Path to a folder containing the CIFs for the database. If this
                is not provided, the default path from DaraSettings will be used.
        """
        self._path_to_cifs = path_to_cifs
        self._preparsed_info: dict = {}

    def get_cifs_by_formulas(
        self,
        formulas: list[str],
        e_hull_filter: float = 0.1,
        copy_files=True,
        dest_dir: str = "dara_cifs",
        exclude_gases: bool = True,
    ):
        """Get a list of database CIF codes corresponding to formulas, and optionally copy CIF
        files into a destination folder.

        Args:
            formulas: List of formulas to search for.
            e_hull_filter: Filter out structures with e_hull above this value (compared
                via spacegroup).
            copy_files: Whether to copy CIF files to a destination folder.
            dest_dir: Destination folder for copied CIF files.
            exclude_gases: Whether to exclude common gases (e.g., O2) from the results.
        """
        file_map = {}
        all_data = []
        for formula in formulas:
            all_data.extend(self.get_formula_data(formula))

        file_map = self._generate_file_map(all_data, e_hull_filter, exclude_gases)

        if copy_files:
            copy_and_rename_files(file_map, dest_dir)

        return [data[1] for data in all_data]

    def get_cifs_by_chemsys(
        self,
        chemsys: str | list[str] | set[str],
        e_hull_filter: float = 0.1,
        copy_files=True,
        dest_dir: str = "dara_cifs",
        exclude_gases: bool = True,
    ):
        """Get a list of database CIF codes corresponding to structures in a chemical system.
        Option to copy CIF files into a destination folder.

        Args:
            chemsys: Chemical system to search for (e.g., "Fe-O").
            e_hull_filter: Filter out structures with e_hull above this value (compared
                via spacegroup).
            copy_files: Whether to copy CIF files to a destination folder.
            dest_dir: Destination folder for copied CIF files.
            exclude_gases: Whether to exclude common gases (e.g., O2) from the results.
        """
        if isinstance(chemsys, str):
            chemsys = chemsys.split("-")

        elements_set = set(chemsys)  # remove duplicate elements
        all_data = []

        for i in range(len(elements_set)):
            for els in itertools.combinations(elements_set, i + 1):
                sub_chemsys = "-".join(sorted(els))
                if sub_chemsys in self.preparsed_info:
                    all_data.extend(self.preparsed_info[sub_chemsys])

        file_map = self._generate_file_map(all_data, e_hull_filter, exclude_gases)

        if copy_files:
            copy_and_rename_files(file_map, dest_dir)

        return [data[1] for data in all_data]

    def get_formula_data(self, formula: str):
        """Get a list of database codes + info corresponding to a formula.

        Args:
            formula: Chemical formula to search for.
        """
        formula_reduced = Composition(formula).reduced_formula
        chemsys = Composition(formula).chemical_system
        db_chemsys = self.preparsed_info.get(chemsys)

        if not db_chemsys:
            logger.warning(f"No database codes found in chemical system: {db_chemsys}!")
            return []

        formula_data = [i for i in db_chemsys if i[0] == formula_reduced]

        if not formula_data:
            logger.warning(f"No database codes found for {formula}!")
            return []

        return formula_data

    def _generate_file_map(
        self,
        all_data,
        e_hull_filter,
        exclude_gases,
        download_folder="dara_downloaded_cifs",
    ):
        """A mapping of database file paths to new file names with structure metadata included."""
        if not self.local_copy_found:
            logger.warning(
                "Local copy of database not found. Attempting to download structures..."
            )
            _ = self.download_structures(
                [data[1] for data in all_data],
                save=True,
                default_folder=download_folder,
            )

        file_map = {}
        for formula, code, sg, e_hull in all_data:
            if exclude_gases and formula in COMMON_GASES:
                logger.info("Skipping common gas: %s", formula)
                continue

            if e_hull is not None and e_hull > e_hull_filter:
                print(
                    f"Skipping high-energy phase: {code} ({formula}, {sg}): e_hull = {e_hull}"
                )
                continue

            e_hull_value = round(1000 * e_hull) if e_hull is not None else None

            fp = (
                f"{self.get_file_path(code)}"
                if self.local_copy_found
                else f"{download_folder}/{code}.cif"
            )
            file_map[fp] = f"{formula}_{sg}_({self.name}_{code})-{e_hull_value}.cif"

        return file_map

    @property
    def path(self) -> Path:
        """Path to local copy of the structure database (i.e., the folder containing all
        the relevant CIFs). Automatically uses the default path if not
        provided.
        """
        p = self._path_to_cifs
        if p is None:
            p = self.default_folder_path

        return Path(p)

    @property
    def local_copy_found(self) -> bool:
        """Check if a local copy of the database is found."""
        return self.path.exists()

    @abstractmethod
    def download_structures(
        self, ids: list[str] | None = None, save=False, default_folder=None
    ) -> list[Cif]:
        """Download structures from the database."""

    @abstractmethod
    def get_file_path(self, cif_id: str | int):
        """Get the path to a CIF file in the database from its database ID."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the database."""

    @property
    @abstractmethod
    def default_folder_path(self) -> Path:
        """Default path to the folder containing the CIFs for the database."""

    @property
    def preparsed_info(self) -> dict:
        """Preparsed information about the database. This is used to quickly filter and
        find structures based on chemical system, formula, etc. This has already been
        generated via the code in the scripts folder.
        """
        return self._preparsed_info


class CODDatabase(StructureDatabase):
    """Class to interact with CIF files acquired from the COD database. This class uses
    a parsed/filtered list of COD numbers that are automatically selected from when doing
    phase searches (data/cod_filtered_info_2024.json.gz).

    This class features a method to download CIFs from the online COD database, as well
    as a method to get CIFs from a local copy of the database. To download the COD
    database in a folder called COD_2024, run the following command (this may take a
    while):

        rsync -av --delete rsync://www.crystallography.net/cif/ COD_2024/
    """

    def __init__(self, path_to_cifs: Path | str | None = None):
        """
        Initialize an object for interacting with CIFs from the ICSD.

        path_to_cifs: Path to a folder containing the CIFs for the database.
        """
        super().__init__(path_to_cifs)
        self._preparsed_info = loadfn(
            Path(__file__).parent / "data/cod_filtered_info_2024.json.gz"
        )

    def download_structures(
        self,
        ids: list[str] | None = None,
        save=False,
        default_folder="downloaded_cod_cifs",
    ) -> list[Cif]:
        """Download structures from the COD database. Note that this downloads directly
        from the COD website, so it may be slow. Please do not abuse this feature.

        Args:
            ids: List of COD IDs to download.
        """
        cifs = thread_map(
            self._download_cod,
            ids,
            chunksize=1,
            max_workers=16,
            desc="Downloading CIFs from COD...",
        )
        if save:
            logger.info(f"Saving downloaded CIFs to {default_folder}")
            default_folder = Path(default_folder)
            if not default_folder.exists():
                default_folder.mkdir()

            for cif in cifs:
                cif.to_file((default_folder / cif.filename).with_suffix(".cif"))

        return cifs

    def get_file_path(self, cif_id: str | int):
        """Get the path to a CIF file in the COD database."""
        cod_id = str(cif_id).ljust(7, "0")
        if len(cod_id) > 7:
            raise ValueError(f"Invalid COD ID provided (too long): {cod_id}")

        return self.path / cod_id[0] / cod_id[1:3] / cod_id[3:5] / f"{cod_id}.cif"

    @property
    def default_folder_path(self) -> Path:
        """Default path to the folder containing the CIFs for the COD database."""
        return PATH_TO_COD

    @property
    def name(self) -> str:
        """Name of the database."""
        return "cod"

    @staticmethod
    def _download_cod(cod_id: str) -> Cif:
        """Download a COD file from its ID."""
        COD_URL = "https://qiserver.ugr.es/cod/{cod_id}.cif"
        # Get the content of the URL
        try:
            url = COD_URL.format(cod_id=cod_id)
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise an error for bad status codes
            with NamedTemporaryFile(mode="w+b") as temp_file:
                temp_file.write(response.content)
                temp_file.flush()
                cif = Cif.from_file(temp_file.name)
            cif.filename = str(cod_id)
        except Exception as e:
            print(f"Failed to download {cod_id}: {e}")
            raise

        return cif


class ICSDDatabase(StructureDatabase):
    """Class to interact with CIF files acquired from the ICSD database. This class uses
    a parsed/filtered list of ICSD_IDs that are automatically selected from when doing
    phase searches (see data/icsd_filtered_info_2024_v2.json.gz). Here we assume you
    have a folder containing relevant downloaded ICSD CIFs located at the
    default path in DaraSettings. You can also configure the path in dara.yaml.

    Each CIF file should be named "icsd_<id>.cif" where <id> is the ICSD code.

    **Legal notice**: to use the ICSD database, you must purchase a paid license that allows
    you to download and keep CIFs as a local copy. We do not provide any CIFs from the
    ICSD and discourage any unpermitted use of the database that is inconsistent
    with your license. Please visit https://icsd.products.fiz-karlsruhe.de/ for more
    information. We are not liable for any misuse of the ICSD database.
    """

    def __init__(self, path_to_cifs: Path | str | None = None):
        """
        Initialize an object for interacting with CIFs from the ICSD.

        :param path_to_icsd: Path to the ICSD database
        """
        super().__init__(path_to_cifs)
        self._preparsed_info = loadfn(
            Path(__file__).parent / "data/icsd_filtered_info_2025_v3.json.gz"
        )

    def get_file_path(self, cif_id: str | int):
        """Get the path to a CIF file in the ICSD database."""
        return self.path / f"icsd_{self._clean_icsd_code(cif_id)}.cif"

    def download_structures(
        self, ids: list[str] | None = None, save=False, default_folder=None
    ) -> list[Cif]:
        """Download structures from the ICSD database."""
        raise NotImplementedError(
            "Downloading from the online ICSD database will not be implemented. Please use a local copy of CIFs"
            " acquired legally through your agreement."
        )

    @property
    def name(self) -> str:
        """Name of the database."""
        return "icsd"

    @property
    def default_folder_path(self) -> Path:
        """Default path to the folder containing the CIFs for the ICSD database."""
        return PATH_TO_ICSD

    @staticmethod
    def _clean_icsd_code(icsd_code):
        """Add leading zeros to the ICSD code."""
        code = str(int(icsd_code))
        return (6 - len(code)) * "0" + code
