# Anil Yildiz
# yildiz@stanford.edu

The .tif files for wind speed are downloaded from:   https://globalwindatlas.info/
The .hgt files for elevation are downloaded from:    https://dds.cr.usgs.gov/srtm/
The .geojson file lists the corners of the rectangular area selected for the data download (Notice that the fifth point is the same as the first point).

The .xyz files are obtained using the software QGIS:
1) Layer -> Add Layer -> Add Raster Layer -> choose .tif file.
2) Raster -> Conversion -> Translate -> new window will pop up.
3) Output file (QGIS 2.x) or Converted file (QGIS 3.x) -> Name -> change the filetype to ASCII Gridded XYZ in the right bottom -> Save.
4) In the console of at the bottom -> make sure it says "gdal_translate -of XYZ".
5) OK -> click OK on any conformation windows that pop up.
