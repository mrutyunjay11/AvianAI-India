"""Geographic species filtering and range derivation for AvianAI India.

Loads regional metadata and regional indices
to filter and prioritize bird species occurring in India and its subregions.
"""

from pathlib import Path
import json
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from config.settings import (
    CONFIG_DIR,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_REGION,
    REGIONAL_INDICES,
    REGIONS_PATH,
)


class GeoFilter:
    """Handles geographic filtering of BirdNET v3.0 species based on official regional metadata."""

    def __init__(self, regions_file: Path = REGIONS_PATH) -> None:
        self.regions_file = Path(regions_file)
        self.regions_data: Dict[str, Any] = {}
        self.regional_indices: Dict[str, List[int]] = {}
        self.pan_india_indices: List[int] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        # Load regions.json if available
        if self.regions_file.exists():
            try:
                with open(self.regions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.regions_data = data.get("regions", {})
            except Exception as e:
                print(f"[WARNING] Failed to parse {self.regions_file}: {e}")
                self.regions_data = {}

        # Load regional index files for India
        pan_india_set: Set[int] = set()

        for region_key, index_path in REGIONAL_INDICES.items():
            if Path(index_path).exists():
                try:
                    with open(index_path, "r", encoding="utf-8") as f:
                        indices = [int(line.strip()) for line in f if line.strip().isdigit()]
                        self.regional_indices[region_key] = indices
                        pan_india_set.update(indices)
                except Exception as e:
                    print(f"[WARNING] Failed to load {index_path}: {e}")
                    self.regional_indices[region_key] = []
            else:
                self.regional_indices[region_key] = []

        self.pan_india_indices = sorted(list(pan_india_set))
        self.regional_indices["pan-india"] = self.pan_india_indices
        self._loaded = True

    def get_region_names(self) -> List[str]:
        """Returns supported region keys."""
        self._ensure_loaded()
        return [
            "pan-india",
            "south-asia-peninsular",
            "indo-gangetic",
            "himalaya",
            "all",
        ]

    def get_region_display_labels(self) -> Dict[str, str]:
        """Returns human-friendly region descriptions."""
        self._ensure_loaded()
        return {
            "pan-india": f"🇮🇳 Pan-India (All Indian Subregions - {len(self.pan_india_indices)} species)",
            "south-asia-peninsular": f"🌴 South India & Sri Lanka ({len(self.regional_indices.get('south-asia-peninsular', []))} species)",
            "indo-gangetic": f"🌾 Indo-Gangetic Plain ({len(self.regional_indices.get('indo-gangetic', []))} species)",
            "himalaya": f"🏔️ Himalaya & Northern Mountain Ecozone ({len(self.regional_indices.get('himalaya', []))} species)",
            "all": "🌍 Global (All 11,560 Species - Unfiltered)",
        }

    def match_region_by_coordinates(self, lat: float, lon: float) -> str:
        """Determines the best matching region for given GPS coordinates."""
        self._ensure_loaded()

        # Check official bounding boxes from regions.json if available
        # Bounding box format in regions.json: [[min_lat, max_lat, min_lon, max_lon]]
        for region_key in ["south-asia-peninsular", "indo-gangetic", "himalaya"]:
            meta = self.regions_data.get(region_key, {})
            bboxes = meta.get("bboxes", [])
            for bbox in bboxes:
                if len(bbox) == 4:
                    min_lat, max_lat, min_lon, max_lon = bbox
                    if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                        return region_key

        # Broad India bounding box: lat 6.0 to 37.5, lon 68.0 to 98.0
        if 6.0 <= lat <= 37.5 and 68.0 <= lon <= 98.0:
            if lat < 21.0:
                return "south-asia-peninsular"
            elif lat < 28.0:
                return "indo-gangetic"
            else:
                return "himalaya"

        # Fallback to Pan-India if near South Asia
        return "pan-india"

    def get_allowed_indices(
        self,
        region: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Optional[List[int]]:
        """Returns the list of allowed species indices based on location or region.

        If region is 'all' or no filtering applies, returns None (meaning all 11,560 allowed).
        """
        self._ensure_loaded()

        target_region = region

        # If lat and lon are provided and region is not explicitly forced, deduce region
        if lat is not None and lon is not None and (target_region is None or target_region == "auto"):
            target_region = self.match_region_by_coordinates(lat, lon)

        if target_region is None:
            target_region = DEFAULT_REGION

        target_region = target_region.lower().strip()

        if target_region in ["all", "global", "none"]:
            return None

        if target_region in self.regional_indices:
            indices = self.regional_indices[target_region]
            if indices:
                return indices

        # Fallback to Pan-India
        return self.pan_india_indices if self.pan_india_indices else None

    def create_species_mask(
        self,
        total_classes: int = 11560,
        region: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> np.ndarray:
        """Returns a boolean mask of shape (total_classes,) where True indicates allowed species."""
        allowed_indices = self.get_allowed_indices(region=region, lat=lat, lon=lon)
        mask = np.zeros(total_classes, dtype=bool)

        if allowed_indices is None:
            mask[:] = True
        else:
            mask[allowed_indices] = True

        return mask

    def filter_probabilities(
        self,
        probabilities: np.ndarray,
        region: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> np.ndarray:
        """Applies geographic mask to an array of probabilities [num_classes] or [batch, num_classes].

        Species outside the geographic zone are set to 0.0, filtering out non-native birds.
        """
        num_classes = probabilities.shape[-1]
        mask = self.create_species_mask(total_classes=num_classes, region=region, lat=lat, lon=lon)

        filtered = probabilities.copy()
        if filtered.ndim == 1:
            filtered[~mask] = 0.0
        elif filtered.ndim == 2:
            filtered[:, ~mask] = 0.0
        return filtered


_GEO_FILTER_INSTANCE: Optional[GeoFilter] = None


def get_geo_filter() -> GeoFilter:
    """Returns singleton instance of GeoFilter."""
    global _GEO_FILTER_INSTANCE
    if _GEO_FILTER_INSTANCE is None:
        _GEO_FILTER_INSTANCE = GeoFilter()
    return _GEO_FILTER_INSTANCE
