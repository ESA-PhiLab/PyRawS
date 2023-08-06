API demonstration 
==================

**Table of contents**

.. contents::
   :local:
   :depth: 1

.. container:: cell markdown

   This notebook is to show and demonstrate the use of Application
   Program Interface (API) developed in the frame of the ``PYRAWS``
   project to open and process ``Sentinel-2 Raw data``, corresponding to
   a decompressed version of `Sentinel-2 L0
   data <https://sentinel.esa.int/documents/247904/685211/sentinel-2-products-specification-document>`__
   with additional metada. The API are demonstrated on the
   ``Temperature Hotspots RAW Sentinel-2 (THRAWS)`` dataset. We will
   introduce the use of the ``Raw_event`` and ``raw_granule`` classes to
   process ``Raw granules`` and ``Raw events`` containing images of
   volcanic eruptions. It will show how to stack different
   ``Raw granules`` acquired during the movement of the satellite along
   track and how to perform a coarse onboard coregistration of ``Raw``
   bands. Furthermore, it will introduce the APIs to extract specific
   bands coordinates. Finally, after introducing the equivalent
   ``L1C_tiles`` and ``L1C_event``, the notebook will show the API to
   mosaic the ``L1C``\ tiles and crop them around the specific
   ``raw_granule`` bands coordinats to have both the ``L1c`` and ``Raw``
   products looking at the same area. Finally, it will show how to
   process the ``L1C`` information to doublechek the presence of an
   eruption by exploiting an algorithm developed on ``L1c`` data that
   would work for ``Raw`` data.

.. container:: cell markdown

   .. rubric:: 1) Imports, paths and variables
      :name: 1-imports-paths-and-variables

.. container:: cell markdown

   Limit CUDA visible devices

.. container:: cell code

   .. code:: python

      import os
      os.environ['CUDA_VISIBLE_DEVICES']='5'

.. container:: cell markdown

   Autoreload

.. container:: cell code

   .. code:: python

      %load_ext autoreload
      %autoreload 2

.. container:: cell markdown

   Imports

.. container:: cell code

   .. code:: python

      import sys
      sys.path.insert(1, os.path.join("..", ".."))
      sys.path.insert(1, os.path.join("..", "..", "scripts_and_studies", "hta_detection_algorithms"))
      from pyraws.raw.raw_event import Raw_event
      from pyraws.l1.l1_event import L1C_event
      from pyraws.utils.l1_utils import read_L1C_image_from_tif
      from pyraws.utils.visualization_utils import plot_img1_vs_img2_bands
      from s2pix_detector import s2pix_detector
      from functools import partial
      from geopy.geocoders import Nominatim
      import geopandas
      from geopandas import GeoSeries
      import matplotlib.pyplot as plt
      from shapely.geometry import Polygon
      from termcolor import colored
      import torch
      from skimage.measure import label, regionprops
      import matplotlib.patches as patches

.. container:: cell markdown

   This import is to remove odd errors on ``libiomp5md.dll``. If you do
   not have them, you can skip it

.. container:: cell code

   .. code:: python

      os.environ['KMP_DUPLICATE_LIB_OK']='True'

.. container:: cell markdown

   Set torch device. Use "CUDA" as default if available.

.. container:: cell code

   .. code:: python

      if torch.cuda.is_available():
          device=torch.device("cuda")
      else:
          device=torch.device("cpu")

.. container:: cell markdown

   Set size of figure plots.

.. container:: cell code

   .. code:: python

      plt.rcParams['figure.figsize'] = [10, 10]

.. container:: cell markdown

   .. rubric:: 2) - Raw_event and raw_granule
      :name: 2---raw_event-and-raw_granule

.. container:: cell markdown

   To request ``Raw`` files it is necessary to query the database by
   specifying a polygon (area) and a date range (start - end). For the
   event ``Etna_00``, shown in the image above, the blue rectangular is
   the polygon used to query the database (the eruption is the blue spot
   in the image in the center of the rectangular). Upon a query, the
   database will download the collection of ``Raw granules`` whose
   reference band (``B02``) intersects the blue polygon in the specified
   date range . An ``Raw granule`` (**red rectangulars**) corresponds to
   the area acquired by the all ``13 Sentinel-2 bands`` **of a single
   detector** over a single acquisition (lasting 3.6 s). The various Raw
   granules in the collection might be produced in different instants
   and by different detectors (Sentinel 2 has 12 detectors staggered
   across track). We named this collection of ``Raw granules`` referred
   to a specific event (``Etna_00``) an ``Raw event``. Such concepts of
   ``Raw granule`` and ``Raw event`` (collection of the
   ``Raw granules``) are made through the classes ``Raw_granule`` and
   ``Raw_event``. When an object ``Raw_event`` is created, it
   instatiates a collection of ``Raw_granule`` objects each one
   containing the information related to each ``Raw granule``.

.. container:: cell markdown

   To create an ``Raw_event`` object, please specify the
   ``requested_bands`` and the requested ``event_name``.

.. container:: cell code

   .. code:: python

      requested_bands=["B8A","B11","B12"]
      event_name="Etna_00"

.. container:: cell markdown

   The next lines will parse query the ``thraw_db.csv`` database with
   the requested ``event_name``, enabling the creation of the
   ``Raw_event`` with the requested bands.

.. container:: cell code

   .. code:: python

      event=Raw_event(device=device)

.. container:: cell code

   .. code:: python

      event.from_database(event_name, requested_bands)

.. container:: cell markdown

   .. rubric:: 3) - Showing Raw granules info
      :name: 3---showing-raw-granules-info

.. container:: cell markdown

   The next lines will show the information related to the granules that
   compose the instantiated ``Raw_event``.

.. container:: cell code

   .. code:: python

      event.show_granules_info()

.. container:: cell markdown

   **Interpretation of granules information.** As you can see, the
   ``Raw_event`` is composed of a collection of ``Raw_granule`` objects,
   matching the ``Raw_granules`` whose reference band interesects the
   area used to request for Raw data. The method
   ``show_granules_info()`` of the class ``Raw_event`` prints all the
   granules composing an **Raw_event**. For each of the granules, the
   function shows the ``granule name``, ``sensing time``,
   ``Creation time``, ``detector number``, ``originality``, ``parents``,
   ``polygon coordinates`` (of vertices), ``cloud coverage``
   (percentage/100). ``originality`` and ``parents`` are needed in case
   the granule is created through some processing of other granules
   (such as stacking or coregistration, see next cells). If this is not
   the case, ``originality`` will be ``True`` and the list of granules
   parents will be empty. If the granule is created by stacking two
   granules, ``originality`` will be ``False`` and ``parents`` will
   contain the name of the granules used for stacking. In this case, all
   the information are also shown for ``parents``.

.. container:: cell markdown

   In the next lines, we will select the granule ``0`` and will show the
   bands requested when the ``Raw_event`` was created.

.. container:: cell code

   .. code:: python

      raw_granule=event.get_granule(0)
      raw_granule.show_bands(downsampling=True)

.. container:: cell markdown

   .. rubric:: 4) - Compose granules
      :name: 4---compose-granules

.. container:: cell markdown

   The APIs offer utils to compose granules along and across track.
   However, the granules from an event cannot composed arbitrarily.
   Indeed, to compose two granules **along track** they must have the
   same **detector number** and the **sensing-time** to be different of
   3.6 s (3 or 4 seconds). For the event ``Etna_00``, granules [0,2],
   [1,3], [3,5] can be stacked along tracks. The next line will stack
   granules [0,2] along track. The string "T" means that the granule 0
   will be stacked on top of 2.

.. container:: cell code

   .. code:: python

      raw_granule_0_2_stacked=event.stack_granules([0,2], "T")

.. container:: cell markdown

   **Showing stacked granule info.** By using the method
   ``get_granule_info()`` of the classs ``raw_granule``, you can get the
   granule information. You can see that granule is now marked as **not
   original** (``originality`` is set to ``False``). This is because the
   ``raw_granule_0_2_stacked`` is the result of combination of two
   granules. In this case, the ``get_granule_info()`` function will show
   will print ``sensing time``, ``acquisition time``,
   ``detector number`` for the granule parents. ``originality`` will be
   ``False`` and the list of granules parents will be not ``None``. You
   can notice that the granule name is composed by the parents'name
   separated by the keyword **STACKED_T**, where **T** means the first
   granule is stacked at the top of the second one.

.. container:: cell code

   .. code:: python

      raw_granule_0_2_stacked.show_granule_info()

.. container:: cell markdown

   The same effect can be used by using the ``Raw_event`` method
   ``get_stackable_granules()``, which permits extracting the couples of
   granules that can be stacked along-track automatically.

.. container:: cell code

   .. code:: python

      stackable_granules, stackable_couples=event.stack_granules_couples()
      raw_granule_0_2_stacked=stackable_granules[0]
      raw_granule_0_2_stacked.show_granule_info()

.. container:: cell markdown

   You can now see by superimposing the bands of stacked granules. As
   you can see that bands do not look coregistered. This is because the
   pushbroom nature of Sentinel-2, for which every band is looking at
   different areas during a single acquisition (granule) (and ``SWIR``
   and ``visible`` bands are respectively rotated).

.. container:: cell code

   .. code:: python

      raw_granule_0_2_stacked.coarse_coregistration(crop_empty_pixels=False, verbose=True).show_bands_superimposition(equalize=False)

.. container:: cell code

   .. code:: python

      raw_granule_0_2_stacked.coarse_coregistration_old(crop_empty_pixels=True, verbose=True).show_bands_superimposition(equalize=False)

.. container:: cell markdown

   .. rubric:: 5) - Coarse bands coregistration
      :name: 5---coarse-bands-coregistration

.. container:: cell markdown

   In some cases, the **event** (*fire/volcanic eruption*) will be not
   contained in a single granule. In other cases, if the eruption is
   located close to the top/bottom margin, the information of some of
   the bands could be missing because of lack of bands registration.
   Therefore, we stack to granules **along track** to try to overcome to
   this limitation.

.. container:: cell markdown

   **Bands can now be roughly coregistered**. Coregistration is perfomed
   by shifting the bands of a number of pixel
   S_{k,l}|_{(S,D)}=\ :math:`[N_{B_k,B_l}, M_{B_k,B_l}]|_{(S,D)}`
   specific for the couple of bands :math:`(B_k,B_l)` produced by the
   detector having detector number :math:`D` in the satellite :math:`S`
   (S2A or S2B). :math:`N_{B_k,B_l}` is the number of along-track shift
   pixels, used to compensate the systematic band shifts due to the
   pushbroom nature of the sensor. Similarly, :math:`M_{B_k,B_l}` is the
   average number of across-track pixels in the ``THRAW`` dataset for a
   certain couple :math:`(S,D)`. To this aim, :math:`S_{k,l}|_{(S,D)}`
   are stored in a ``Look Up Table`` and used regardelss the position of
   the satellite. It shall be noted that :math:`S_{k,l}|_{(S,D)}`
   indicates the number of pixels shift for which the band :math:`B_l`
   shall be moved to match the band :math:`B_k`. Since :math:`(B_k,B_l)`
   could have different resolution, :math:`S_{k,l}|_{(S,D)}` is
   expressed with respect to :math:`B_l` resolution. Having more than 2
   bands leads to coregister al the bands with respect to the first one.
   For instance, when using [``B8A``, ``B11``,\ ``B12``] bands ``B12``
   and ``B11`` are coregistered with respect to ``B8A``.

.. container:: cell markdown

   The next line extracts the granule 2 from the event and performs the
   coregistration.

.. container:: cell code

   .. code:: python

      # Get granule 2
      raw_granule=event.get_granule(2)
      # Get granule 2 from the Raw event and perform coarse coregistration
      raw_granule_registered=event.coarse_coregistration(granules_idx=[2])

.. container:: cell markdown

   Showing unregistered vs coarse registered granule.

.. container:: cell code

   .. code:: python

      # Plotting unregistered vs registered granule
      fig, ax = plt.subplots(1,2)
      ax[0].set_title("Unregistered")
      raw_granule_tensor=raw_granule.as_tensor(downsampling=True)
      # Plot normalizing on the max
      ax[0].imshow(raw_granule_tensor/raw_granule_tensor.max())
      ax[1].set_title("Coarse coregistered")
      raw_granule_registered_tensor=raw_granule_registered.as_tensor(downsampling=True)
      # Plot normalizing on the max
      ax[1].imshow(raw_granule_registered_tensor/raw_granule_registered_tensor.max())
      plt.show()

.. container:: cell markdown

   As you can see in the image above, on the bottom of the registered
   images there is an image where only one band is not null. This is due
   to the fact that ``B11`` and ``B12`` are shifted to match ``B8A``,
   leaving some area uncovered. This could create a problem every time
   the **high temperature anomaly** (*fire/volcanic eruption*) will be
   placed in that area. To overcome to this limitation, it is possible
   to fill the missing pixels by filling the missing pixels with the
   ones of adjacent granules. To this aim, you can call the
   ``coarse_coregistration`` function with the flag
   ``use_complementary_granules`` set to ``True``. In this way, if
   appropriate adjacent filler granules are available, they will be
   retrieved and adjacent pixels used to fill the missing pixels.
   FIlling will be performed across-track as well.

.. container:: cell code

   .. code:: python

      # Get granule 2 from the Raw event and perform coarse coregistration with filling elements
      raw_granule_registered_filled=event.coarse_coregistration(granules_idx=[2], use_complementary_granules=True)

.. container:: cell markdown

   As you can see in the image below, missing pixels are now filled.

.. container:: cell code

   .. code:: python

      # Plotting unregistered vs registered granule
      fig, ax = plt.subplots(1,2)
      ax[0].set_title("Unregistered")
      raw_granule_tensor=raw_granule.as_tensor(downsampling=True)
      # Plot normalizing on the max
      ax[0].imshow(raw_granule_tensor/raw_granule_tensor.max())
      ax[1].set_title("Coarse coregistered (filled)")
      raw_granule_registered_filled_tensor=raw_granule_registered_filled.as_tensor(downsampling=True)
      # Plot normalizing on the max
      ax[1].imshow(raw_granule_registered_filled_tensor/raw_granule_registered_filled_tensor.max())

.. container:: cell markdown

   Alternatively, it is also possible to crop those empty pixels. To
   this aim, you can call the ``coarse_coregistration`` function with
   the flag ``crop_empty_pixels`` set to ``True``. If you set both the
   ``use_complementary_granules`` and ``crop_empty_pixels`` flags to
   ``True``, filler elements will be used when available, otherwise
   missing pixels will be cropped.

.. container:: cell code

   .. code:: python

      # Get granule 2 from the L0 event and perform coarse coregistration with filling elements
      raw_granule_registered_cropped=event.coarse_coregistration(granules_idx=[2], crop_empty_pixels=True)

.. container:: cell markdown

   The next plot shows all the possible cases.

.. container:: cell code

   .. code:: python

      raw_granule_registered_cropped_tensor=raw_granule_registered_cropped.as_tensor(downsampling=True)
      fig, ax = plt.subplots(2,2)
      ax[0,0].set_title("Unregistered")
      ax[0,0].imshow(raw_granule_tensor/raw_granule_tensor.max())
      ax[0,1].set_title("Coarse coregistered")
      # Plot normalizing on the max
      ax[0,1].imshow(raw_granule_registered_tensor/raw_granule_registered_tensor.max())
      ax[1,0].set_title("Coarse coregistered (filled)")
      ax[1,0].imshow(raw_granule_registered_filled_tensor/raw_granule_registered_filled_tensor.max())
      ax[1,1].set_title("Coarse coregistered (cropped)")
      ax[1,1].imshow(raw_granule_registered_cropped_tensor/raw_granule_registered_cropped_tensor.max())
      plt.show()

.. container:: cell markdown

   .. rubric:: 6) - Getting Raw granule and bands coordinates
      :name: 6---getting-raw-granule-and-bands-coordinates

.. container:: cell markdown

   **Get image coordinates.**

.. container:: cell markdown

   Granules coordinates are stored ``InventoryMetadata.xml``. The
   ploygon represents the ``Granule Footprint``, meaning the area
   covered by all the bands of one detector. The next lines extract the
   granule 2 from the Raw_event and performs the coregistration with
   filling elements. Then, it extract the coordinates of the entire
   granule.

.. container:: cell code

   .. code:: python

      raw_granule_coregistered=event.coarse_coregistration(granules_idx=[2], use_complementary_granules=True, downsampling=False)
      coordinates=raw_granule_coregistered.get_granule_coordinates()

.. container:: cell markdown

   The next lines show the area covered by the granule.

.. container:: cell code

   .. code:: python

      def plot_granule_coordinates(coordinates):
          try:
              world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
              world=world.to_crs("EPSG:4326")
              geolocator = Nominatim(user_agent="google")
              reverse = partial(geolocator.reverse, language="en")
              address=reverse(coordinates[0])[0]
              country=address[-address[::-1].find(","):][1:]
              coordinates=[(y,x) for (x,y) in coordinates]
              poly=GeoSeries([Polygon([x for x in coordinates +[coordinates[0]]])])
              poly=poly.set_crs("EPSG:4326")
              ax=world.query('name == \"'+country+'\"').plot()
              poly.plot(facecolor='red', edgecolor='red',ax=ax)
              print("Address: ", colored(address, "red"))
          except:
              print("Impossible to plot granule over the requested area.")
              
      plot_granule_coordinates(coordinates)

.. container:: cell markdown

   Moreover, it is also possible to perform georeferencing of the single
   bands. The next lines shows how to get the coordinates of the
   coregistered bands.

.. container:: cell code

   .. code:: python

      band_shifted_dict=raw_granule_coregistered.get_bands_coordinates()
      band_shifted_dict

.. container:: cell markdown

   Showing bands coregistered bands positions.

.. container:: cell code

   .. code:: python

      def plot_granule_coordinates(coordinates, coordinates_shifted_dict):
          world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
          world=world.to_crs("EPSG:4326")
          geolocator = Nominatim(user_agent="google")
          reverse = partial(geolocator.reverse, language="en")
          address=reverse(coordinates[0])[0]
          country=address[-address[::-1].find(","):][1:]
          coordinates=[(y,x) for (x,y) in coordinates]
          poly=GeoSeries([Polygon([x for x in coordinates +[coordinates[0]]])])
          poly=poly.set_crs("EPSG:4326")
          band_names=list(coordinates_shifted_dict.keys())
          color_mask=['blue', 'yellow', 'green']
          ax=world.query('name == \"'+country+'\"').plot()
          poly.plot(facecolor='red', edgecolor='red',ax=ax)
          
          for n in range(len(coordinates_shifted_dict)): 
              coordinates_shifted_band=coordinates_shifted_dict[band_names[n]]
              address_shifted=reverse(coordinates_shifted_dict[band_names[n]][0])[0]
              country_shifted=address_shifted[-address_shifted[::-1].find(","):][1:]
              coordinates_shifted_band=[(y,x) for (x,y) in coordinates_shifted_band]
              poly_shifted=GeoSeries([Polygon([x for x in coordinates_shifted_band +[coordinates_shifted_band[0]]])])
              poly_shifted.plot(facecolor=color_mask[n%3], edgecolor=color_mask[n%3],ax=ax)

          print("Address: ", colored(address, "red"))
              
      plot_granule_coordinates(coordinates,band_shifted_dict)

.. container:: cell markdown

   To compare the coordinates of your the different bands, you can use
   the ``create_geee_polygon`` function and use the polygons on
   ``Google Earth Engine`` and visually compare them with respect the
   bands images and with respect to the polygon area.

.. container:: cell code

   .. code:: python

      def create_geee_polygon(coordinates, polygon_name, switch_lat_lon=True):
          if switch_lat_lon:
              pol_coordinates=[[y,x] for (x,y) in coordinates]
          else:
              pol_coordinates=coordinates
          pol_string="var "+polygon_name + " = ee.Geometry.Polygon("+str(pol_coordinates)+");"
          print(pol_string)
          return pol_string

.. container:: cell markdown

   .. rubric:: 7) - Open and crop L1C tiles
      :name: 7---open-and-crop-l1c-tiles

.. container:: cell markdown

   .. rubric:: 7.a) - Open L1C tiles and events
      :name: 7a---open-l1c-tiles-and-events

.. container:: cell markdown

   Getting ``L1C`` event corresponding to the ``L0`` event.

.. container:: cell code

   .. code:: python

      l1c_event=L1C_event(device=device)
      l1c_event.from_database(event_name, requested_bands)

.. container:: cell markdown

   Printing ``L1C`` tiles info.

.. container:: cell code

   .. code:: python

      l1c_event.show_tiles_info()

.. container:: cell markdown

   .. rubric:: 7.b) Mosaicing and cropping L1C tiles on an L0 band
      coordinates.
      :name: 7b-mosaicing-and-cropping-l1c-tiles-on-an-l0-band-coordinates

.. container:: cell markdown

   It is now possible to mosaic L1C tiles and crop them over the area of
   the previous L0 granules. In this way, it is possible to process L1C
   tiles and reproject the retrieved information (e.g., event bounding
   boxes) on the correspondent L0 granule. The cropped L1C file is a
   "TIF" file. An ``ending`` is added to keep track of the indices of
   the granules used(i.e. ending = "2").

.. container:: cell code

   .. code:: python

      ending="2"
      output_cropped_tile_path=l1c_event.crop_tile(band_shifted_dict[requested_bands[0]], None,out_name_ending=ending, lat_lon_format=True)

.. container:: cell code

   .. code:: python

      output_cropped_tile_path

.. container:: cell markdown

   Plotting results.

.. container:: cell markdown

   **Raw granule**

.. container:: cell code

   .. code:: python

      raw_granule_coregistered.show_bands_superimposition()

.. container:: cell markdown

   **L1C cropped tile**

.. container:: cell code

   .. code:: python

      l1c_tif, coords_dict, expected_class=read_L1C_image_from_tif(event_name, ending, device=device)
      plt.imshow(l1c_tif)

.. container:: cell code

   .. code:: python

      raw_granule_coregistered.show_bands_superimposition()

.. container:: cell markdown

   .. rubric:: 8) - Processing L1C tiles
      :name: 8---processing-l1c-tiles

.. container:: cell markdown

   It is now possible to process the cropped ``L1C`` data to spot a
   volcanic eruption by using a simplified version of the algorithm
   `Massimetti, Francesco, et
   al. <https://www.mdpi.com/2072-4292/12/5/820>`__.

.. container:: cell markdown

   Get **hotmap**.

.. container:: cell code

   .. code:: python

      _, l1c_filtered_alert_matrix, l1c_alert_matrix=s2pix_detector(l1c_tif)

.. container:: cell markdown

   Show **alert matrix** and **filtered_alert_map**.

.. container:: cell code

   .. code:: python

      plot_img1_vs_img2_bands(l1c_tif, l1c_tif, ["alert_map", "filtered_alert_map"], l1c_alert_matrix, l1c_filtered_alert_matrix)

.. container:: cell markdown

   Show **Raw granule** and equivalent **L1C cropped area** with the
   correspondent **filtere_alert_map**.

.. container:: cell code

   .. code:: python

      raw_granule_tensor=raw_granule_coregistered.as_tensor()
      plot_img1_vs_img2_bands(raw_granule_tensor/raw_granule_tensor.max(), l1c_tif, ["L0", "L1C + Hotmap"], None, l1c_filtered_alert_matrix)

.. container:: cell markdown

   Show **L1C cropped area** with the correspondent
   **filtere_alert_map** and **bounding boxes**.

.. container:: cell code

   .. code:: python

      mask=l1c_filtered_alert_matrix.numpy()
      lbl = label(mask)
      props = regionprops(lbl)
      l1c_numpy=l1c_tif.numpy()
      fig, ax = plt.subplots()
      ax.imshow(l1c_tif)
      for prop in props:
           print('Found bbox', prop.bbox)
           bbox = prop.bbox         #   x,       y,        width, height
           rect = patches.Rectangle((bbox[1], bbox[0]), abs(bbox[1]-bbox[3]), abs(bbox[0]-bbox[2]), linewidth=2, edgecolor='y', facecolor='none')
           ax.add_patch(rect)
      plt.show()
