import pyogrio

from sggain.gis.io import configure_large_geojson_reads


def test_configure_large_geojson_reads_removes_gdal_object_size_limit():
    pyogrio.set_gdal_config_options({"OGR_GEOJSON_MAX_OBJ_SIZE": None})

    configure_large_geojson_reads()

    assert pyogrio.get_gdal_config_option("OGR_GEOJSON_MAX_OBJ_SIZE") in {"0", 0}
