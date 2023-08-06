Glossary
========

**Table of contents**

.. contents::
   :local:
   :depth: 1

.. _coarse-coregistration:

Coarse Coregistration
---------------------
Lightweight spatial coregistration method optimized for onboard-satellite applications. It merely shifts the various bands by a fixed factor depending on the bands, the satellite, and the detector number.

.. _sentinel-2-l0-data:

Sentinel-2 L0 Data
------------------
These are data transmitted to the ground from Sentinel-2 satellites. The L0 format is compressed to minimize downlink bandwidth requirements. 

.. _sentinel-2-raw-data:

Sentinel-2 Raw Data
-------------------
In this project, Sentinel-2 Raw data is a decompressed version of Sentinel-2 L0 data with additional metadata produced on the ground. These data closely emulate the ones produced by Sentinel-2 detectors, excluding the effects due to compression and onboard equalization, which are not compensated at this stage. Sentinel-2 raw data is the data used in this project. 

.. _sentinel-2-raw-granule:

Sentinel-2 Raw Granule
----------------------
A granule is an image captured by a Sentinel-2 detector during a single acquisition lasting 3.6 s. Granules are defined at L0 level. However, processing performed on the ground does not alter the image content, apart from the decompression process, but merely provides additional metadata.

.. _sentinel-2-raw-event:

Sentinel-2 Raw Event
--------------------
To download L0 data, it is necessary to specify a polygon that surrounds a specific area of interest. This results in the download of all Sentinel-2 Raw granules whose reference band intersects the specified polygon, forming a Raw event. 

.. _sentinel-2-l1c-data:

Sentinel-2 L1C Data
-------------------
The Level 1-C (L1C) is one format for Sentinel-2 data. To convert Sentinel-2 Raw data to L1C data, numerous processing steps are applied to correct defects, including band coregistration, ortho-rectification, decompression, noise suppression, and others.

.. _sentinel-2-l1c-event:

Sentinel-2 L1C Event
--------------------
This is similar to a Sentinel-2 Raw event but applied to Sentinel-2 L1C data.

.. _sentinel-2-l1c-tile:

Sentinel-2 L1C Tile
-------------------
The Sentinel-2 L1C tile is the smallest L1C product that can be downloaded.

.. _raw-complementary-granule:

Raw Complementary Granule
-------------------------
Given a certain set of bands of interest `[Bx,By,...,Bz]`, Raw complementary granules are the granules adjacent to Raw-useful-granules that can be used to fill missing pixels of `[By,...,Bz]` bands due to their coregistration with respect to the band `Bx`.

.. _raw-useful-granule:

Raw Useful Granule
------------------
Given a certain set of bands of interest `[Bx,By,...,Bz]`, where `Bx` is the first band in the set, a Raw useful granule is one of the collection of Sentinel-2 Raw granules that compose a Sentinel-2 Raw event whose band `Bx` includes (or intersects) a certain area of interest (e.g., an eruption or an area covered by a fire).

.. _raw-events-n-granules:

.. admonition:: Raw Events and Raw Granules
   :class: note

   .. figure:: _static/etna_00_granules.png
      :alt: Etna_00 Granules

      The footprint of all Sentinel-2 Raw granules downloaded for the eruption named "Etna_00".

   When downloading Sentinel-2 Raw data, a polygon surrounding the area of interest and a date must be specified. Considering the unique imaging properties of the Sentinel-2 sensor, data bands at the Raw level do not necessarily capture the same area. 
   To ensure the capture of all bands relevant to a specific event (such as volcanic eruptions or wildfires), rectangular polygons of 28x10 :math:`km^2` are used, centered on the events. This results in the download of all Raw granules whose reference band (B02) intersects with the polygon area.
   The provided image illustrates all the Sentinel-2 Raw granules that have been downloaded for the eruption labelled as "Etna_00" using a white rectangular polygon. For our purposes, a "Sentinel-2 Raw event" is defined as the collection of Raw granules downloaded for each corresponding entry in our database.
   However, as can be observed in the image, the majority of the Raw granules within the "Etna_00" event do not contain the volcanic eruption of interest (marked by a large red spot). Indeed, only the sections depicted by the yellow and pink rectangles intersect with or include parts of the volcanic eruption.
   Additionally, even if a Raw granule intersects or includes an event, this does not guarantee that all bands within that Raw granule will intersect or include the event. Specifically, we deem a granule to be 'useful' if its B8A band intersects with the event. This is the case for the yellow rectangle, but not for the pink one. After co-registration, the other bands are adjusted to match the area of B8A.
   Finally, in certain cases, parts of the event (such as eruptions or wildfires) could extend to the top or bottom edge of the polygon. In these cases, some bands might be missing for a portion of the area of interest. To ensure complete coverage, it's crucial to consider complementary granules, which serve to fill in the missing sections of the area of interest.
   For each "Sentinel-2 Raw event", our THRAWS dataset clearly identifies which Raw granules are deemed 'useful' or 'complementary'. However, the entire collection of Raw granules is provided for each event, enabling users interested in using other bands to detect warm temperature anomalies to do so.

