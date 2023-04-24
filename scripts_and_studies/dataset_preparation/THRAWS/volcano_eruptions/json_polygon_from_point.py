import geopandas
from shapely.geometry import Polygon
import argparse
#from l0_dataset_preparation.volcano_eruptions.create_volcano_csv import create_rectangular_polygon, parse_csv
from .create_volcano_csv import create_rectangular_polygon, parse_csv



def create_polygon_from_coordinates(lat, lon, center_distance_x, center_distance_y, json_name=None):
    center_coordinates={'lat' : lat, 'lon' : lon}
    polygon_dict=create_rectangular_polygon(center_coordinates, center_distance_x, center_distance_y)
    polygon=Polygon([polygon_dict['SW'], polygon_dict['NW'], polygon_dict['NE'], polygon_dict['SE']])
    json_content=geopandas.GeoSeries([polygon]).to_json()

    if json_name is not None:
        with open(json_name, 'w') as f:
            f.write(json_content)
            f.write('\n\n')
            f.write(str(polygon_dict['SW']))
            f.write('\n\n')
            f.write(str(polygon_dict['NW']))
            f.write('\n\n')
            f.write(str(polygon_dict['NE']))
            f.write('\n\n')
            f.write(str(polygon_dict['SE']))

    return polygon

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat', type=float, help='Point latitude.')
    parser.add_argument('--lon', type=float, help='Point latitude.')
    parser.add_argument('--dx_km', type=float, help='Horizontal distance from the polygon center in km.', default=5.0)
    parser.add_argument('--dy_km', type=float, help='Vertical distance from the polygon center in km.', default=14.0)
    parser.add_argument('--json_name', type=str, help='json file name.', default="my_json_polygon.json")
    pargs=parser.parse_args()
    create_polygon_from_coordinates(pargs.lat, pargs.lon, pargs.dx_km * 1000, pargs.dy_km * 1000, pargs.json_name)

if __name__ == "__main__":
    main()