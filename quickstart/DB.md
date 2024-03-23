## Working with Database

This section of the respository contains an example of database (`THRAWS`) that is used by PyRawS to read and process [Sentinel-2 Raw data](#sentinel-2-raw-data) and [Sentinel-2 L1 data](#sentinel-2-l1-data) correctly. The minimal fields of the database include:

> [!WARNING]
> Additionally, make sure that all required dependencies and packages are installed before running the code in the notebook. Please note that it is important to carefully follow the instructions in the notebook to ensure that a database is created correctly and without errors. 





| ID_event              | Start       | End               | Sat | Coords (Lat, Lon)    | class    | Raw_useful_granules | Raw_complementary_granules | Polygon_square                                                                                                                                                                      | Raw_files                                                                                                                                                                                                                                                                                                                                                                                                                                                        | l1c_files                                       | bbox_list                                 | bbox_list_merged                                     | Polygon                                                                                                                                                                        |
|-----------------------|-------------|-------------------|-----|----------------------|----------|---------------------|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|--------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Barren_Island_00      | 28/09/2018  | 28/09/2018 23:59  | S2A | (12.28474241, 93.86212046) | eruption | [2]                 | [4]                       | POLYGON ((93.82076157486179 12.244061101857305, 93.8207488665122 12.325417348895849, 93.9034920534878 12.325417348895849, 93.90347934513821 12.244061101857305, 93.82076157486179 12.244061101857305)) | ['S2A_OPER_MSI_L0__GR_EPAE_20180929T132957_S20180928T042450_D10_N02.06', 'S2A_OPER_MSI_L0__GR_EPAE_20180929T132957_S20180928T042453_D09_N02.06', 'S2A_OPER_MSI_L0__GR_EPAE_20180929T132957_S20180928T042453_D10_N02.06', 'S2A_OPER_MSI_L0__GR_EPAE_20180929T132957_S20180928T042457_D09_N02.06', 'S2A_OPER_MSI_L0__GR_EPAE_20180929T132957_S20180928T042457_D10_N02.06'] | ['S2A_MSIL1C_20180928T040541_N0206_R047_T46PEU_20180929T135308'] | {2: [[[403, 290], [427, 284], [435, 316], [410, 322]]]} | {2: [[[403, 290], [427, 284], [435, 316], [410, 322], [403, 290]]]} | POLYGON((93.81618097217948 12.158183811523221, 93.81613704194822 12.411292126632908, 93.90810387805179 12.41129212663291, 93.90805994782052 12.158183811523221, 93.81618097217948 12.158183811523221)) |
| Barren_Island_01      | 28/10/2018  | 28/10/2018 23:59  | S2A | (12.28474241, 93.86212046) | eruption | [2]                 | [4]                       | POLYGON ((93.82076157486179 12.244061101857305, 93.8207488665122 12.325417348895849, 93.9034920534878 12.325417348895849, 93.90347934513821 12.244061101857305, 93.82076157486179 12.244061101857305)) | ['S2A_OPER_MSI_L0__GR_EPAE_20181028T064658_S20181028T042451_D10_N02.06', 'S2A_OPER_MSI_L0__GR_EPAE_20181028T064658_S20181028T042455_D09_N02.06', 'S2A_OPER_MSI_L0__GR_EPAE_20181028T064658_S20181028T042455_D10_N02.06', 'S2A_OPER_MSI_L0__GR_EPAE_20181028T064658_S20181028T042458_D09_N02.06', 'S2A_OPER_MSI_L0__GR_EPAE_20181028T064658_S20181028T042458_D10_N02.06'] | ['S2A_MSIL1C_20181028T040851_N0206_R047_T46PEU_20181028T070450'] | {2: [[[391, 59], [412, 54], [418, 





* **ID_event**:	ID of the event (e.g., volcanic-eruption, wildfire, not-event). All the other fields of the row are referred to that `Sentinel-2` acquisition.
* **class**:	class of the event (e.g., eruption, fire, not-event). Leave it **empty**
* **Raw_useful_granules**:		list of [Raw useful granules](#raw-useful-granule). Set to `None` or leave it empty if you do not know what are the [Raw useful granules](#raw-useful-granule).
* **Raw_complementary_granules**:	list of [Raw complementry granules](#raw-complementary-granule). Set to `None` or leave it empty if you do not know what are the [Raw complementry granules](#raw-complementary-granule).
* **Raw_files**:	list of [Raw granules](#sentinel-2-raw-granule) (**mandatory**).
* **l1c_files**: list of [L1 tiles](#sentinel-2-l1c-tile) (mandatory if you need L1C data).
* **bbox_list**:	dictionary {[Raw useful granules](#raw-useful-granule) : [bounding box list for that granule]}.  Set to `None` or leave it **empty** if you do not know the bounding box location.

To create a new database (e.g., `my_database_name`), please, proceed as follows:

1. Create a ".csv" file with the structure shown above and place it into the `database`subfloders (e.g., `my_db.csv`). You can use start from the [thraws_db.csv](https://github.com/ESA-PhiLab/PyRawS/-/blob/main/PyRawS/database/thraws_db.csv) database and edit it accordingly to your specification.
2. Create subdirectory `my_database_name` in the `data` subdirectory and populate it with the corresponding [Sentinel-2 Raw data](#sentinel-2-raw-data) and [Sentinel-2 L1 data](#sentinel-2-l1-data) as described in [Data directory](#data-directory).
3. Update the `DATABASE_FILE_DICTIONARY` in `PyRawS/utils/constants.py` as follows:

```DATABASE_FILE_DICTIONARY={"THRAWS" : "thraws_db.csv", "my_database_name" : "my_db.csv"}```

> [!IMPORTANT]
> In case you want to create your own database of event for your target aplication, the user should refer to the notebook called "database_creation.ipynb". This notebook contains the necessary code and instructions for creating the database. Simply follow the steps outlined in the notebook to successfully create something similar to the "THRAWS" database.

> [!TIP]
> **N.B** The creation of a database is not mandatory. However, it is strongly advisable. Indeed, without creating a database you can still open `Raw data` as described in [Open a Raw event from path](#open-a-raw-event-from-path). However, some pieces of information such as the [Raw useful granules](#raw-useful-granule) associated to a specific event, the event bounding boxes or the image class can be retrieved only when the database is set-up.