Quickstart
==================

**Table of contents**

.. contents::
   :local:
   :depth: 1

The next examples (with the exception of :ref:`open-a-raw-event-from-path`) will exploit the `THRAWS` database to showcase the use of PyRawS, but they are applicable to any databases compatible with PyRawS :ref:`database`.

.. _open-a-raw-event-from-path:

1) Open a Raw event from path
-------------------------------

The next code snipped will showcase how to use PyRawS to open a :ref:`Raw_event <sentinel-2-raw-event>` `My_Raw_data`, included in the `THRAWS` database by using its `PATH`.

.. note::

    We assume to have the `data` directory in the same directory where you execute the code snippet below.

To manipulate `Raw events`, `PyRawS` offer a class called `Raw_event`. To open an event, we will use the :ref:`Raw_event <sentinel-2-raw-event>` class method `from_path(...)`, which parses the database specified, retrieves the event specified by `id_event` and opens it with the requested bands (`bands_list`).
When you open an event, you can specify which bands to use. If `bands_list` is not specified, the method `from_path(...)` will return all the bands.

.. code-block:: python

    from pyraws.raw.raw_event import Raw_event
    #Instantiate an empty Raw_event
    raw_event=Raw_event()

    #Bands to open.
    bands_list=["B8A", "B11", "B12"]

    #Read "Etna_00"  from THRAWS
    raw_event.from_path(#Path to the Etna_00 event
                        raw_dir_path="Path_to_my_RAW_data",
                        #Bands to open. Leave to None to use all the bands.
                        bands_list=bands_list,
                        #If True, verbose mode is on.
                        verbose=True)

The example above can be used directly, **even if you did not set up a** `PyRawS database`. However, `PyRawS` offer some API to parse directly `Raw_event` parsing a database. with no need to specify a path. Please, check `Open a Raw event from database`.

.. raw:: html

   <br />
   <br />
   <br />

.. _open-a-raw-event-from-database:

2) Open a Raw event from database
-----------------------------------

The next code snipped will showcase how to use PyRawS to open the `Raw_event` `Etna_00` included in the `THRAWS` database. To do that,
To manipulate `Raw events` objects, `PyRawS` will exploit the `Raw_event` class method `from_database(...)`, which parses the associated `.csv` file located in `PyRawS/database` with no need to specify the `PATH` from the user. To execute the next code snipped, we assume that you have already downloaded and set-up the `THRAWS` database as specified in `databases compatible with PyRawS`.

As for the method `from_path(...)` described in `Open a Raw event from path`, you can specify which bands to use. If `bands_list` is not specified, the method `from_database(...)` will return all the bands.

.. code-block:: python

    from pyraws.raw.raw_event import Raw_event
    #Instantiate an empty Raw_event
    raw_event=Raw_event()

    #Bands to open.
    bands_list=["B8A", "B11", "B12"]

    #Read "Etna_00"  from THRAWS
    raw_event.from_database(#Database ID_EVENT
                          id_event="Etna_00",
                          #Bands to open. Leave to None to use all the bands.
                          bands_list=bands_list,
                          #If True, verbose mode is on.
                          verbose=True,
                          #Database name
                          database="THRAWS")

.. warning::
   All the next examples will assume you already have downloaded and set-up the `THRAWS` database as specified in `databases compatible with PyRawS. However, they can work by using `from_path(...)` instead of `from_database(...)` and specifying the `PATH` to the `Etna_00` event manually.

.. raw:: html

   <br />
   <br />
   <br />

.. _show-raw-event-information:

3) Show Raw Granules Information of a Raw Event
--------------------------------------------------

As specified in :ref:`sentinel-2-raw-event`, a Sentinel-2 Raw event is a collection of Sentinel-2 Raw granules. As for :ref:`sentinel-2-raw-event`, Sentinel-2 Raw granules are modeled in PyRawS through a dedicated class `Raw_granule`.

The following code snippet will show how to get the information about the Sentinel-2 Raw granules that compose the `Etna_00` Raw event. The class method `show_granules_info()` will print the list of events and some metadata for each event. To get the same information as a dictionary `{granule name : granule info}` for easy manipulation, you can use the `Raw_event` class method `get_granules_info(...)`.

.. code-block:: python

   from pyraws.raw.raw_event import Raw_event

   # Instantiate an empty Raw_event
   raw_event = Raw_event()

   # Bands to open.
   bands_list = ["B8A", "B11", "B12"]

   # Read "Etna_00" from THRAWS database.
   # You can also read it by using raw_event.from_path(...).
   raw_event.from_database(
       # Database ID_EVENT
       id_event="Etna_00",
       # Bands to open. Leave to None to use all the bands.
       bands_list=bands_list,
       # If True, verbose mode is on.
       verbose=True,
       # Database name
       database="THRAWS"
   )

   # Printing granules info
   raw_event.show_granules_info()

   # Getting Raw granules info dictionary {granule name : granule info}
   granules_info_dict = raw_event.get_granules_info()


4) Get a Single Raw Granule from a Raw Event
---------------------------------------------

The class `Raw_event` contains a list of objects `Raw_granule`, each one modeling a specific Sentinel-2 Raw granule that belongs to that Sentinel-2 Raw event (please, check :ref:`sentinel-2-raw-event` and :ref:`sentinel-2-raw-granule` for more information).

The different `Raw_granule` objects are sorted alphabetically and are accessible through indices. The following code snippet will show how to get a specific Sentinel-2 Raw granule by using the `Raw_event` class method `get_granule(granule_idx)`, where `granule_idx` is the granule index. The function returns a `Raw_granule` object. As for `Raw_event` objects, it is possible to print or retrieve metadata information for a specific `Raw_granule` by using the `Raw_granule` methods `get_granule_info()` and `show_granule_info()`.

.. code-block:: python

   from pyraws.raw.raw_event import Raw_event

   # Instantiate an empty Raw_event
   raw_event = Raw_event()

   # Bands to open.
   bands_list = ["B8A", "B11", "B12"]

   # Read "Etna_00" from THRAWS database.
   # You can also read it by using raw_event.from_path(...).
   raw_event.from_database(
       # Database ID_EVENT
       id_event="Etna_00",
       # Bands to open. Leave to None to use all the bands.
       bands_list=bands_list,
       # If True, verbose mode is on.
       verbose=True,
       # Database name
       database="THRAWS"
   )

   # Read the granule 0 of the Etna_00 event.
   raw_granule_0 = raw_event.get_granule(0)

   # Printing the info of the granule 0
   raw_granule_0.show_granule_info()

   # Getting Raw granules info dictionary {granule name : granule info}
   granule_0_info_dict = raw_granule_0.get_granule_info()


5) Access a Raw Granule Pixels
--------------------------------

To visualize the values of a `Raw_data` object, it is possible to return it as a PyTorch tensor. However, since the different bands have different resolutions, depending on the bands that we want to shape as a tensor, it is necessary to upsample/downsample some of them to adapt them to the band with higher/smaller resolution. The following code snippet will open the `Etna_00` with bands `B02` (10 m), `B8A` (20 m), `B11` (20 m), get the granule with index 1, and will return the first two bands in the collection as tensor by performing upsample.

.. code-block:: python

   from pyraws.raw.raw_event import Raw_event

   # Instantiate an empty Raw_event
   raw_event = Raw_event()

   # Bands to open.
   bands_list = ["B04", "B8A", "B11"]

   # Read "Etna_00" from THRAWS database.
   # You can also read it by using raw_event.from_path(...).
   raw_event.from_database(
       # Database ID_EVENT
       id_event="Etna_00",
       # Bands to open. Leave to None to use all the bands.
       bands_list=bands_list,
       # If True, verbose mode is on.
       verbose=True,
       # Database name
       database="THRAWS"
   )

   # Read the granule 1 of the Etna_00 event.
   raw_granule_1 = raw_event.get_granule(1)

   # Returning the bands B04 and B8A of raw_granule_1 as tensor by upsampling.
   raw_granule_1_tensor = raw_granule_1.as_tensor(
       # List of bands to transform as tensor
       requested_bands=["B04", "B8A"],
       # Set to True to perform downsampling (default)
       downsampling=False
   )


6) Superimpose Raw Granule Bands
-----------------------------------

It is possible to superimpose `Raw_granule` bands by using the class method `show_bands_superimposition(...)`. (N.B. it is possible to superimpose up to three bands).
In case bands have different resolution, you need to specify if you want to superimpose them by performing downsampling (default) or upsampling. The following code snippet will open the `Etna_00` with bands `B02`, `B8A`, `B11`, `B12`, get the granule with index 0, and will superimpose the last three bands.

.. code-block:: python

   from pyraws.raw.raw_event import Raw_event
   import matplotlib.pyplot as plt

   # Instantiate an empty Raw_event
   raw_event = Raw_event()

   # Bands to open.
   bands_list = ["B04", "B8A", "B11", "B12"]

   # Read "Etna_00" from THRAWS database.
   # You can also read it by using raw_event.from_path(...).
   raw_event.from_database(
       # Database ID_EVENT
       id_event="Etna_00",
       # Bands to open. Leave to None to use all the bands.
       bands_list=bands_list,
       # If True, verbose mode is on.
       verbose=True,
       # Database name
       database="THRAWS"
   )

   # Read the granule 0 of the Etna_00 event.
   raw_granule_0 = raw_event.get_granule(0)

   # Superimpose the last three bands of raw_granule_0.
   raw_granule_0.show_bands_superimposition(
       # Bands to superimpose
       requested_bands=["B04", "B11", "B12"],
       # Set to True to perform downsampling
       downsampling=True
   )
   plt.show()

The previous code snippet will display the image below. As you can see, the various bands of the image lack of coregistration of the various bands.

.. figure:: _static/granule_superimposition.png
   :alt: Alt Text
   :align: center

.. raw:: html

   <br />
   <br />
   <br />


7) How to Perform Coarse Coregistration of Raw Granules
--------------------------------------------------------

PyRawS offers some utils to perform `coarse coregistration`_ on `Raw_granule` objects. You can coregister a specific `Raw_granule` object of the `Raw_event` collection by calling the `coarse_coregistration(...)` method of the `Raw_event` class by selecting the correspondent index through the `granules_idx` input.

.. code-block:: python

   from pyraws.raw.raw_event import Raw_event

   # Instantiate an empty Raw_event
   raw_event = Raw_event()

   # Bands to open.
   bands_list = ["B04", "B8A", "B11"]

   # Read "Etna_00" from THRAWS database.
   # You can also read it by using raw_event.from_path(...).
   raw_event.from_database(
       # Database ID_EVENT
       id_event="Etna_00",
       # Bands to open. Leave to None to use all the bands.
       bands_list=bands_list,
       # If True, verbose mode is on.
       verbose=True,
       # Database name
       database="THRAWS"
   )

   # Perform the coarse coregistration of the "Etna_00" event.
   # Missing pixels will be filled with zeros.
   raw_coreg_granule_2 = raw_event.coarse_coregistration(
       # Granule index to coregister.
       granules_idx=[2]
   )

.. _coarse coregistration: https://pyraws.readthedocs.io/en/latest/coarse_coregistration.html



8) Coarse Coregistration with Cropped Missing Pixels
------------------------------------------------------

The previous code snippet returns the coarse coregistration of the granule 2 of the "Etna_00" event. The coarse coregistration is performed by shifting the bands `B8A` and `B11` with respect to the band `B04`, which is the first in the collection. The missing pixels produced by the shift of the bands `B8A` and `B11` will be filled by zeros. The superimposition of the coregistered bands with zero-filling is shown in the image below ("coregistration").

It is possible to crop the missing values by setting the argument `crop_empty_pixels=True`, as in the snippet below.

.. code-block:: python

   from pyraws.raw.raw_event import Raw_event

   # Instantiate an empty Raw_event
   raw_event = Raw_event()

   # Bands to open.
   bands_list = ["B04", "B8A", "B11"]

   # Read "Etna_00" from THRAWS database.
   # You can also read it by using raw_event.from_path(...).
   raw_event.from_database(
       # Database ID_EVENT
       id_event="Etna_00",
       # Bands to open. Leave to None to use all the bands.
       bands_list=bands_list,
       # If True, verbose mode is on.
       verbose=True,
       # Database name
       database="THRAWS"
   )

   # Perform the coarse coregistration of the "Etna_00" event.
   # Missing pixels will be cropped.
   raw_coreg_granule_0_with_crop = raw_event.coarse_coregistration(
       # Granule index to coregister.
       granules_idx=[2],
       # Cropping missing pixels.
       crop_empty_pixels=True
   )


9) Coarse Coregistration with Filling Elements
-----------------------------------------------

Alternatively, you can fill the missing pixels with filler elements taken from other `Raw_granule` objects when available. This is done by setting the argument `use_complementary_granules=True`. In this case, the compatibility of adjacent `Raw_granule` objects will be checked by the `coarse_coregistration(...)` API and use it in case it is available.

When filling `Raw_granule` objects are not available, missing pixels will be cropped if `crop_empty_pixels` is set to True. The superimposition of the coregistered bands with crop is shown in the image below ("coregistration with crop").

.. code-block:: python

   from pyraws.raw.raw_event import Raw_event

   # Instantiate an empty Raw_event
   raw_event = Raw_event()

   # Bands to open.
   bands_list = ["B04", "B8A", "B11"]

   # Read "Etna_00" from THRAWS database.
   # You can also read it by using raw_event.from_path(...).
   raw_event.from_database(
       # Database ID_EVENT
       id_event="Etna_00",
       # Bands to open. Leave to None to use all the bands.
       bands_list=bands_list,
       # If True, verbose mode is on.
       verbose=True,
       # Database name
       database="THRAWS"
   )

   # Perform the coarse coregistration of the "Etna_00" event.
   # Missing pixels will be cropped.
   raw_coreg_granule_0_with_fill = raw_event.coarse_coregistration(
       # Granule index to coregister.
       granules_idx=[2],
       # Search for filling elements among adjacent Raw granules.
       use_complementary_granules=True,
       # Cropping missing pixels when compatible Raw granules are not available.
       crop_empty_pixels=True
   )

The superimposition of the coregistered bands with filling elements is shown in the image below ("coregistration with fill").

.. figure:: _static/coregistration.png
   :alt: Alt Text
   :align: center


10) How to Get the Coordinates of a Raw Granule Band
-----------------------------------------------------

It is possible to get coordinates of the vertices of a `Raw_granule` object. Georeferencing is performed by using the information of the bands shift used to perform the `coarse coregistration` with respect to the band `B02` and by exploiting the coordinates of the `Raw granule` footprint (please, refer to [Sentinel-2 Products Specification Document](https://sentinel.esa.int/documents/247904/685211/sentinel-2-products-specification-document)).

The code snippet below shows how to get the information of the different bands of a `Raw granule`.

.. code-block:: python

   from pyraws.raw.raw_event import Raw_event

   # Instantiate an empty Raw_event
   raw_event = Raw_event()

   # Bands to open.
   bands_list = ["B04", "B8A", "B11"]

   # Read "Etna_00" from THRAWS database.
   # You can also read it by using raw_event.from_path(...).
   raw_event.from_database(
       # Database ID_EVENT
       id_event="Etna_00",
       # Bands to open. Leave to None to use all the bands.
       bands_list=bands_list,
       # If True, verbose mode is on.
       verbose=True,
       # Database name
       database="THRAWS"
   )

   # Read the granule 1 of the Etna_00 event.
   raw_granule_1 = raw_event.get_granule(1)

   # Get bands coordinates.
   bands_coordinates_dict = raw_granule_1.get_bands_coordinates()

The code snippet above returns a `bands_coordinates_dict`, a dictionary structured as `{band_name : [BOTTOM-LEFT(lat, lon),  BOTTOM-RIGHT(lat, lon), TOP-RIGHT(lat, lon), TOP-LEFT(lat, lon)]}`.


11) Raw_event: Database Metadata
----------------------------------

This example will show you how to extract database metadata associated to an `Raw event`. To run the next example it is necessary to have set up database how described in :ref:`database`.

Database metadata include:

- Event class (e.g., `eruption`, `fire`, `not_event`)
- List of `Raw useful granules`
- `{Useful granule : Bounding box}` dictionary

.. code-block:: python

   from pyraws.raw.raw_event import Raw_event

   # Instantiate an empty Raw_event
   raw_event = Raw_event()

   # Bands to open.
   bands_list = ["B04", "B8A", "B11"]

   # Read "Etna_00" from THRAWS database.
   # raw_event.from_path(...) cannot be used.
   raw_event.from_database(
       # Database ID_EVENT
       id_event="Etna_00",
       # Bands to open. Leave to None to use all the bands.
       bands_list=bands_list,
       # If True, verbose mode is on.
       verbose=True,
       # Database name
       database="THRAWS"
   )

   # Extract event class.
   raw_event.get_event_class()

   # Extract indices of Raw useful granules.
   raw_event.get_useful_granules_idx()

   # Get {Useful granule : Bounding box} dictionary.
   raw_event.get_bounding_box_dict()


12) Export a Raw_granule to TIF
---------------------------------

This example will show how to export a `Raw_granule` to [TIF](https://en.wikipedia.org/wiki/TIFF) files. To do this, you need to provide the path to a target directory, which will contain a TIF file for each band.

.. code-block:: python

   from pyraws.raw.raw_event import Raw_event

   # Instantiate an empty Raw_event
   raw_event = Raw_event()

   # Bands to open.
   bands_list = ["B04", "B8A", "B11"]

   # Read "Etna_00" from THRAWS database.
   # raw_event.from_path(...) cannot be used.
   raw_event.from_database(
       # Database ID_EVENT
       id_event="Etna_00",
       # Bands to open. Leave to None to use all the bands.
       bands_list=bands_list,
       # If True, verbose mode is on.
       verbose=True,
       # Database name
       database="THRAWS"
   )

   # Apply coarse coregistration to the Raw granule with index 0 and return it.
   raw_granule_0 = raw_event.coarse_coregistration([0])

   # Save folder path.
   output_tif_folder = "raw_target_folder"

   # Export a TIF file for each band.
   raw_granule_0.export_to_tif(save_path=output_tif_folder)

