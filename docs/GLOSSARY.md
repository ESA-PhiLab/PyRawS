
## Glossary
* ### Coarse coregistration
  Lightweight spatial coregistration method optimized for onboard-satellite applications. It simply shifts the various bands of a fixed factor that depends only on the bands, the satellite and detector number.

* ### Sentinel-2 L0 data
  Sentinel-2 data at `level-0` (`L0`) are data that are transmitted to Ground from Sentinel-2 satellites. The `L0` format is compressed to diminish downlink bandwidth requirements. For more information, refer to the [Sentinel-2 Products Specification Document](https://sentinel.esa.int/documents/247904/685211/sentinel-2-products-specification-document)

* ### Sentinel-2 Raw data
  In the frame of this project, the [Sentinel-2 Raw](#sentinel-2-raw-data) represents a particular product in the Sentinel-2 processing chain that matches a  decompressed version of [Sentinel-2 L0 data](sentinel-2-l0-data) with additional metadata that are produced on ground. Once decompressed, `Sentinel-2 Raw data` are the data available on Ground that better emulate the one produced by Sentinel-2 detectors with the exception of the effects due to compression and onboard equalization, which are not compensated at this stage. Therefore, `Sentinel-2 raw data` are those exploited in this project. For more information, refer to the [Sentinel-2 Products Specification Document](https://sentinel.esa.int/documents/247904/685211/sentinel-2-products-specification-document). <br>
**N.B**: the nomenclature ```raw data``` and its location in the Sentinel-2 processing chain is specific for this project only.

* ### Sentinel-2 Raw granule
    A `granule` is the image acquired by a Sentinel-2 detector during a single acquisition lasting 3.6 s. Granules are defined at [L0](#sentinel-2-raw-data) level. However, since the processing perfomed on the ground between L0 and raw data does not alter the image content (with the exception of the decompression process) but just provide additional metadata, granules are defined also at [Sentinel-2 Raw](#sentinel-2-raw-data) level.
    Given the pushbroom nature of the Sentinel-2 sensor, bands do not look at the same area at [Raw](#sentinel-2-raw-data) level. For more information, refer to the [Sentinel-2 Products Specification Document](https://sentinel.esa.int/documents/247904/685211/sentinel-2-products-specification-document)

* ### Sentinel-2 Raw event
   [Sentinel-2 Raw data](#sentinel-2-raw-data) are produced by decompressing [Sentinel-2 L0 data](sentinel-2-raw-data). To download L0 data,  it is necessary to specify one polygon that surrounds a particular area-of-interest. This leads to download all those [Sentinel-2 Raw granules](#sentinel-2-raw-granule) whose reference band intersects the specified polygon. Such collection is a `Raw-event`. Each `Raw-event` matches one of the `ID_event` entry of the database. <br>
    For each `Raw-event`, we do not provide all the collection of  [Sentinel-2 Raw granules](#sentinel-2-Raw-granule), but only the set of [Raw data useful granules](#raw-data-useful-granule) and [Raw data complementary granules](#Raw-data-complementary-granule). For an intuitive example, please, check [Raw events and granules](#raw-events-and-raw-granules).

* ### Sentinel-2 L1C data
  The `Level 1-C` (`L1C`) is one format for `Sentinel-2` data. To convert [Sentinel-2 Raw data](#sentinel-2-raw-data) to `L1C` data, numerous processing steps are applied to correct defects, including bands coregistration, ortho-rectification, decompression, noise-suppression and other. For more information, refer to the [Sentinel-2 Products Specification Document](https://sentinel.esa.int/documents/247904/685211/sentinel-2-products-specification-document).

* ### Sentinel-2 L1C event
  Same concept for [Sentinel-2 Raw events](#sentinel-2-raw-event) but applied on [Sentinel-2 L1C data](#sentinel-2-l1c-data).

* ### Sentinel-2 L1C tile
  The `Sentinel-2 L1C tile` is the minimum `L1C` product that can be downloaded.

* ### Raw complementary granule
  Given a certain set of bands of interest `[Bx,By,...,Bz]`, `Raw complementarey granules` are the granules adjacents at [Raw-useful-granules](#raw-useful-granule) that that can be used to fill missing pixels of `[By,...,Bz]` bands due to their coregistration with respecto the band `Bx`. For an intuitive example, please, check [Raw events and granules](#raw-events-and-raw-granules).

* ### Raw useful granule
  Given a certain set of bands of interest `[Bx,By,...,Bz]`, where `Bx` is the first band in the set, an `Raw useful granule` is one of the collection of [Sentinel-2 Raw granules](#sentinel-2-raw-granule) that compose a [Sentinel-2 Raw event](#sentinel-2-raw-event) whose band `Bx` include (or intersects) a certain area of interest (e.g., an eruption or an area covered by a fire). For an intuitive example, please, check [Raw data events and granules](#raw-events-and-raw-granules).