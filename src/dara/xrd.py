"""Load and process XRD data files (.xrdml, .xy, .rasx)."""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xmltodict
from dict2xml import dict2xml
from monty.json import MSONable

from dara.utils import get_logger

logger = get_logger(__name__)

class XRDData(MSONable):
    """General XRD data class; this is the base class for XRDMLFile, XYFile and other
    XRD data formats. This class ensures that all XRD data can be serialized.
    """

    def __init__(
        self,
        angles: list | np.ndarray,
        intensities: list | np.ndarray,
        errors: list | np.ndarray | None = None,
    ):
        """Initialize XRD data from angles (2-theta values) and intensities/counts."""
        self._angles = np.array(angles)
        self._intensities = np.array(intensities)
        self._errors = np.array(errors) if errors is not None else None

    @property
    def angles(self) -> np.ndarray:
        """2-theta values."""
        return self._angles

    @property
    def intensities(self) -> np.ndarray:
        """Intensity values (counts)."""
        return self._intensities

    @property
    def errors(self) -> np.ndarray | None:
        """Errors in intensity values."""
        return self._errors

    def plot(self, style="line", ax=None, **kwargs):
        """Plot XRD data.

        Args:
            ax: existing matplotlib axis to plot on
            style: either "points" or "line"
            kwargs: keyword arguments to pass to matplotlib.pyplot.plot

        Returns
        -------
            matplotlib axis
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        if style == "points":
            ax.plot(self.angles, self.intensities, "+", ls="", ms=2, **kwargs)
        elif style == "line":
            ax.plot(self.angles, self.intensities, lw=1, **kwargs)
        else:
            raise ValueError(f"Invalid style {style}")

        ax.set_xlabel(r"$2\theta$ (deg)")
        ax.set_ylabel("Intensity (counts)")
        return ax

    @classmethod
    def from_file(cls, path: str | Path):
        """Load data from file. To be implemented in subclasses."""
        raise NotImplementedError

    def to_xy_file(self, fn: str | Path = "xrd_data.xy") -> None:
        """Save as a .xy file.

        Args:
            fn: filename to save to. Defaults to "xrd_data.xy".

        Returns
        -------
            filename of saved file
        """
        if self.errors is not None:
            np.savetxt(
                Path(fn).as_posix(),
                np.column_stack((self.angles, self.intensities, self.errors)),
                fmt="%f",
            )
        else:
            np.savetxt(
                Path(fn).as_posix(),
                np.column_stack((self.angles, self.intensities)),
                fmt="%f",
            )


class RawFile(XRDData):
    """Load Rigaku RAW file data format."""

    def __init__(self, angles, intensities, binary_data: bytes | None = None):
        super().__init__(angles, intensities)
        self._binary_data = binary_data

    @classmethod
    def from_file(cls, path: str | Path) -> RawFile:
        """Load data from a raw file."""
        path = Path(path)
        (angles, intensities), binary_data = load_raw(path)
        return cls(angles, intensities, binary_data)

    @property
    def binary_data(self) -> bytes | None:
        """Binary data."""
        return self._binary_data

    def to_raw_file(self, fn: str | Path = "xrd_data.raw") -> None:
        """Save as a raw file.

        Args:
            fn: filename to save to. Defaults to "xrd_data.raw".
        """
        with open(Path(fn), "wb") as f:
            f.write(self.binary_data)


class XRDMLFile(XRDData):
    """XRDML file class, useful for loading .xrdml data. This is the file type used by
    the Aeris instrument.
    """

    def __init__(self, angles, intensities, xrdml_dict: dict | None = None):
        """Initialize an XRDMLFile object; providing dictionary allows one to serialize
        and deserialize the XRDML file.
        """
        super().__init__(angles, intensities)
        self._xrdml_dict = xrdml_dict

    @property
    def xrdml_dict(self) -> dict | None:
        """Dictionary representation of the XRDML file."""
        return self._xrdml_dict

    @classmethod
    def from_file(cls, path: str | Path) -> XRDMLFile:
        """Load data from an XRDML file."""
        xrdml_dict = load_xrdml(Path(path))
        angles, intensities = get_xrdml_data(xrdml_dict)
        return cls(angles=angles, intensities=intensities, xrdml_dict=xrdml_dict)

    def to_xrdml_file(self, fn: str | Path = "xrd_data.xrdml") -> None:
        """Save as an XRDML file.

        Args:
            fn: filename to save to. Defaults to "xrd_data.xrdml".
        """
        with open(Path(fn), "w") as f:
            f.write(dict2xml(self.xrdml_dict))


class XYFile(XRDData):
    """XY file class, useful for loading .xy data."""

    def __init__(self, angles, intensities, errors: list | np.ndarray | None = None):
        super().__init__(angles, intensities, errors)

    @classmethod
    def from_file(cls, path: str | Path) -> XYFile:
        """Load data from a .xy file."""
        path = Path(path)
        try:
            data = np.loadtxt(path, unpack=True)
        except ValueError:
            # Try to load the file with a comma delimiter
            data = np.loadtxt(path, unpack=True, delimiter=",")
        if len(data) == 2:
            angles, intensities = data
            errors = None
        else:
            angles, intensities, errors = data  # if it is xye data
        return cls(angles, intensities, errors)


class RASXFile(XRDData):
    """Load Rigaku RASX file data format."""

    def __init__(self, angles, intensities, binary_data: bytes | None = None):
        super().__init__(angles, intensities)
        self._binary_data = binary_data

    @classmethod
    def from_file(cls, path: str | Path) -> RASXFile:
        """Load data from a RASX file."""
        path = Path(path)
        (angles, intensities), binary_data = load_rasx(path)
        return cls(angles, intensities, binary_data)

    @property
    def binary_data(self) -> bytes | None:
        """Binary data."""
        return self._binary_data

    def to_rasx_file(self, fn: str | Path = "xrd_data.rasx") -> None:
        """Save as a RASX file.

        Args:
            fn: filename to save to. Defaults to "xrd_data.rasx".
        """
        with open(Path(fn), "wb") as f:
            f.write(self.binary_data)


def load_xrdml(file: Path) -> dict:
    """Load an XRDML file and returns a dictionary using xmltodict."""
    with file.open("r", encoding="utf-8") as f:
        return xmltodict.parse(f.read())


def get_xrdml_data(xrd_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Get angles and intensities from an XRDML dictionary."""
    min_angle = float(
        xrd_dict["xrdMeasurements"]["xrdMeasurement"]["scan"]["dataPoints"][
            "positions"
        ][0]["startPosition"]
    )
    max_angle = float(
        xrd_dict["xrdMeasurements"]["xrdMeasurement"]["scan"]["dataPoints"][
            "positions"
        ][0]["endPosition"]
    )

    intensities = xrd_dict["xrdMeasurements"]["xrdMeasurement"]["scan"]["dataPoints"][
        "counts"
    ]["#text"]
    intensities = np.array([float(val) for val in intensities.split()])
    angles = np.linspace(min_angle, max_angle, len(intensities))
    return angles, intensities


def xrdml2xy(fn: str | Path, target_folder: Path = None) -> Path:
    """Convert .xrdml file to .xy file (and save)."""
    fn = Path(fn)
    if target_folder is None:
        target_folder = fn.parent
    target_path = target_folder / fn.with_suffix(".xy").name

    XRDMLFile.from_file(fn).to_xy_file(target_path)
    return target_path


def hex2float(hex_string: bytes) -> float:
    # Assuming hex_string is a bytes object representing a float in hex
    return struct.unpack("f", hex_string)[0]


def hex2int(hex_string: bytes) -> int:
    # Assuming hex_string is a bytes object representing an int in hex
    return struct.unpack("i", hex_string)[0]


def load_raw(file: Path | str) -> tuple[tuple[np.ndarray, np.ndarray], bytes]:
    """Convert raw file to xy data."""
    with open(file, "rb") as f:
        content = f.read()
    size_float = 4  # Assuming 4 bytes for a float
    size_int = 4  # Assuming 4 bytes for an int

    # Extracting start angle, end angle, and step size from binary content
    start_ang = hex2float(content[2962 : 2962 + size_float])
    end_ang = hex2float(content[2966 : 2966 + size_float])
    count = hex2int(content[3154 : 3154 + size_int])
    start_idx = 3158
    angles = np.zeros(count)
    intensities = np.zeros(count)

    for j in range(count):
        ang = start_ang + (j / (count - 1)) * (end_ang - start_ang)
        its = hex2float(
            content[start_idx + j * size_float : start_idx + (j + 1) * size_float]
        )
        angles[j] = ang
        intensities[j] = its

    return (angles, intensities), content


def raw2xy(fn: str | Path, target_folder: Path = None) -> Path:
    """Convert .raw file to .xy file (and save)."""
    fn = Path(fn)
    if target_folder is None:
        target_folder = fn.parent
    target_path = target_folder / fn.with_suffix(".xy").name

    RawFile.from_file(fn).to_xy_file(target_path)
    return target_path


def load_rasx(file: Path | str) -> tuple[tuple[np.ndarray, np.ndarray], bytes]:
    """Convert RASX file to xy data.

    RASX files are ZIP archives containing:
    - root.xml file
    - Data*/ folders with Profile*.txt files (scan_angle, intensity, attenuation)
    - MesurementConditions*.xml files (note: typo in original format)

    This function extracts the first scan's data from the first Profile file found.

    Returns:
        Tuple of ((angles, intensities), content) where content is the raw ZIP bytes.
        The content is kept for potential round-trip saving, though it's rarely used.
    """
    file = Path(file)
    with open(file, "rb") as f:
        content = f.read()

    with zipfile.ZipFile(io.BytesIO(content), "r") as zip_ref:
        namelist = zip_ref.namelist()

        if "root.xml" not in namelist:
            raise ValueError("No XML file found in the RASX archive.")

        # Find Data folders
        data_folders: dict[str, list[str]] = {}
        for name in namelist:
            if name.startswith("Data") and "/" in name:
                folder = name.split("/")[0]
                if folder not in data_folders:
                    data_folders[folder] = []
                data_folders[folder].append(name)

        # Get the first Profile file from the first Data folder
        profile_path = None
        for _folder_name, file_paths in sorted(data_folders.items()):
            profile_paths = [p for p in file_paths if "Profile" in p]
            if profile_paths:
                if len(profile_paths) > 1:
                    log_msg = (
                        f"Multiple Profile files found in the RASX archive. Using the first one: {profile_paths[0]}"
                    )
                    logger.warning(log_msg)
                profile_path = profile_paths[0]
                break

        if profile_path is None:
            raise ValueError("No Profile file found in the RASX archive.")

        # Read profile data (three columns: scan_angle, intensity, attenuation)
        # Using numpy.loadtxt to handle whitespace-separated values
        # Decode with utf-8-sig to handle BOM if present
        with zip_ref.open(profile_path) as profile_file:
            # Read as text and decode with utf-8-sig to remove BOM if present
            text_content = profile_file.read().decode("utf-8-sig")
            # Use StringIO to create a file-like object for np.loadtxt
            profile_data = np.loadtxt(io.StringIO(text_content), unpack=False)

        # Extract angles (first column) and intensities (second column)
        # Third column is attenuation, which we don't need for now
        if profile_data.ndim == 1:
            # Single row case
            angles = np.array([profile_data[0]])
            intensities = np.array([profile_data[1]])
        else:
            angles = profile_data[:, 0]
            intensities = profile_data[:, 1]

    return (angles, intensities), content


def rasx2xy(fn: str | Path, target_folder: Path = None) -> Path:
    """Convert .rasx file to .xy file (and save)."""
    fn = Path(fn)
    if target_folder is None:
        target_folder = fn.parent
    target_path = target_folder / fn.with_suffix(".xy").name

    RASXFile.from_file(fn).to_xy_file(target_path)
    return target_path


# Extension-to-class mapping for all supported XRD formats.
# Add new formats here to make them available everywhere automatically.
_FORMAT_LOADERS: dict[str, type[XRDData]] = {
    ".xy": XYFile,
    ".xye": XYFile,
    ".txt": XYFile,
    ".scn": XYFile,
    ".xrdml": XRDMLFile,
    ".raw": RawFile,
    ".rasx": RASXFile,
}


def load_pattern(path: str | Path) -> XRDData:
    """Load an XRD pattern from any supported file format.

    Supported formats: .xy, .xye, .txt, .scn, .xrdml, .raw, .rasx.

    Args:
        path: the path to the pattern file.

    Returns:
        an XRDData instance (XYFile, XRDMLFile, RawFile, or RASXFile).

    Raises:
        ValueError: if the file extension is not supported.
    """
    path = Path(path)
    ext = path.suffix.lower()
    loader = _FORMAT_LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported pattern file format: '{ext}'. "
            f"Supported formats: {', '.join(sorted(_FORMAT_LOADERS.keys()))}"
        )
    return loader.from_file(path)


def convert_pattern_to_xy(
    fn: str | Path, target_folder: Path | None = None
) -> Path:
    """Convert any supported XRD pattern file to .xy format.

    If the file is already in .xy format, it is copied to the target folder.

    Args:
        fn: the path to the pattern file.
        target_folder: the folder to save the .xy file in. Defaults to the
            same folder as the input file.

    Returns:
        the path to the generated .xy file.
    """
    import shutil

    fn = Path(fn)
    if target_folder is None:
        target_folder = fn.parent
    target_path = target_folder / fn.with_suffix(".xy").name

    ext = fn.suffix.lower()
    if ext in (".xy", ".xye", ".txt", ".scn"):
        if fn.resolve() != target_path.resolve():
            shutil.copy2(fn, target_path)
    else:
        load_pattern(fn).to_xy_file(target_path)

    return target_path
