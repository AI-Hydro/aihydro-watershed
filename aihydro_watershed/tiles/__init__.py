"""Map-tile builders (basemap slice B2). Offline: reads staged data, writes under ``<AIHYDRO_HOME>/merit/tiles``."""
from aihydro_watershed.tiles.merit_rivers import AREA_BANDS, build_merit_river_tiles, tile_paths

__all__ = ["AREA_BANDS", "build_merit_river_tiles", "tile_paths"]
