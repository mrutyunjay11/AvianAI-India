"""AvianAI India Species Catalog Parser and Resolver.

Parses the 11,560 classes (Scientific name_Common name),
maps indices to species structures, and handles formatting for UI and CLI.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from config.settings import LABELS_PATH


@dataclass(frozen=True)
class BirdSpecies:
    """Represents a bird species classification entry."""

    index: int
    raw_label: str
    scientific_name: str
    common_name: str
    emoji: str = ""

    @property
    def display_name(self) -> str:
        return self.common_name if self.common_name else self.scientific_name

    @property
    def full_display(self) -> str:
        if self.common_name and self.scientific_name:
            return f"{self.common_name} ({self.scientific_name})"
        return self.raw_label


class LabelCatalog:
    """Manages the 11,560 official BirdNET v3.0 species labels."""

    def __init__(self, labels_file: Union[str, Path] = LABELS_PATH) -> None:
        self.labels_file = Path(labels_file)
        self.species_by_index: Dict[int, BirdSpecies] = {}
        self.index_by_scientific: Dict[str, int] = {}
        self.index_by_common: Dict[str, int] = {}
        self.species_list: List[BirdSpecies] = []
        self._loaded = False

        if self.labels_file.exists():
            self._load()

    def _load(self) -> None:
        """Parses the labels text file."""
        with open(self.labels_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.species_list.clear()
        self.species_by_index.clear()
        self.index_by_scientific.clear()
        self.index_by_common.clear()

        for idx, line in enumerate(lines):
            # Format is typically "Scientific name_Common name"
            if "_" in line:
                parts = line.split("_", 1)
                scientific = parts[0].strip()
                common = parts[1].strip()
            else:
                scientific = line
                common = line

            species = BirdSpecies(
                index=idx,
                raw_label=line,
                scientific_name=scientific,
                common_name=common,
                emoji="",
            )

            self.species_list.append(species)
            self.species_by_index[idx] = species
            self.index_by_scientific[scientific.lower()] = idx
            self.index_by_common[common.lower()] = idx

        self._loaded = True

    def __len__(self) -> int:
        self.ensure_loaded()
        return len(self.species_list)

    def ensure_loaded(self) -> None:
        """Ensures labels are parsed."""
        if not self._loaded:
            if not self.labels_file.exists():
                raise FileNotFoundError(f"Labels catalog file not found: {self.labels_file}")
            self._load()

    def get_by_index(self, index: int) -> BirdSpecies:
        """Retrieves species info by output tensor index."""
        self.ensure_loaded()
        if index in self.species_by_index:
            return self.species_by_index[index]
        return BirdSpecies(
            index=index,
            raw_label=f"Unknown_{index}",
            scientific_name=f"Species_{index}",
            common_name=f"Species {index}",
            emoji="",
        )

    def find_index(self, name: str) -> Optional[int]:
        """Finds index by common or scientific name."""
        self.ensure_loaded()
        q = name.lower()
        if q in self.index_by_common:
            return self.index_by_common[q]
        if q in self.index_by_scientific:
            return self.index_by_scientific[q]
        for sp in self.species_list:
            if q in sp.common_name.lower() or q in sp.scientific_name.lower():
                return sp.index
        return None

    def search_by_common(self, query: str) -> List[BirdSpecies]:
        """Finds species whose common name matches query."""
        self.ensure_loaded()
        q = query.lower()
        return [sp for sp in self.species_list if q in sp.common_name.lower()]

    def search_by_scientific(self, query: str) -> List[BirdSpecies]:
        """Finds species whose scientific name matches query."""
        self.ensure_loaded()
        q = query.lower()
        return [sp for sp in self.species_list if q in sp.scientific_name.lower()]

    @property
    def total_classes(self) -> int:
        self.ensure_loaded()
        return len(self.species_list)


_CATALOG_INSTANCE: Optional[LabelCatalog] = None


def get_label_catalog(labels_file: Union[str, Path] = LABELS_PATH) -> LabelCatalog:
    """Returns singleton instance of LabelCatalog."""
    global _CATALOG_INSTANCE
    if _CATALOG_INSTANCE is None:
        _CATALOG_INSTANCE = LabelCatalog(labels_file=labels_file)
    return _CATALOG_INSTANCE
