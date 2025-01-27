<img src="images/vias_logo.svg">

# VIAS

[![Python 3.11](https://img.shields.io/badge/python-3.11-red.svg)](https://www.python.org/downloads/release/python-3119/)
[![CI/CD Pipeline](https://github.com/NikHoh/VIAS/actions/workflows/ci-cd-pipeline.yml/badge.svg)](https://github.com/NikHoh/VIAS/actions/workflows/ci-cd-pipeline.yml)
[![Static Badge](https://img.shields.io/badge/DOI-10.1109%2FOJITS.2023.3299496-yellow)](https://ieeexplore.ieee.org/document/10196046)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview

**This is the inital commit's README. [More updates](#upcoming-features) in the next two to three weeks.**

VIAS (Versatile Intelligent Aerial Streets) is a many-objective path planning framework designed to optimize UAV flight paths in real-world urban environments. Given a start and goal coordinate, along with the operational space's origin and dimensions, VIAS can handle customizable objective and constraint functions to evaluate path quality.

The framework computes a [Pareto set](https://en.wikipedia.org/wiki/Multi-objective_optimization) (a set of optimal compromise solutions) of paths, enabling users to make informed decisions about which path to use.

### Provided Example Objectives
- Minimize **risk** to city residents in case of UAV malfunctions.
- Minimize **noise** exposure for city residents.
- Minimize **radio signal disturbance** to UAV connectivity.
- Minimize **energy consumption** along the aerial corridor.

### Provided Example Constraints
- Ensure compliance with **minimum and maximum flight altitudes**.
- **Avoid collisions** with static obstacles.

### Data Sources
The data used to calculate objective functions and constraints is georeferenced and derived from:
- [OpenStreetMap (OSM)](https://www.openstreetmap.org)
- [OpenCelliD (OCID)](https://www.opencellid.org)

### Scientific Foundation
This code is based on theories and methodologies from the following publications:

1. **[Three-dimensional urban path planning for aerial vehicles regarding many objectives]**  
  _Authors: N. Hohmann, S. Brulin, J. Adamy, and M. Olhofer_
  _Journal/Conference: [IEEE Open Journal of Intelligent Transportation Systems]_  
  _DOI/Link: [10.1109/OJITS.2023.3299496](https://ieeexplore.ieee.org/document/10196046)_

2. **[Multi-objective 3D path planning for UAVs in large-scale urban scenarios]**  
  _Authors: N. Hohmann, M. Bujny, J. Adamy, and M. Olhofer_
  _Conference: [2022 IEEE Congress on Evolutionary Computation (CEC)]_  
  _DOI/Link: [10.1109/CEC55065.2022.9870265](https://ieeexplore.ieee.org/document/9870265)_

3. **[Three-dimensional Many-objective Path Planning and Traffic Network Optimization for Urban Air Mobility Applications Under Social Considerations]**
  _Author: N. Hohmann_
  _Type: Dissertation_
  _DOI/Link: [10.26083/tuprints-00028839](https://tuprints.ulb.tu-darmstadt.de/28839/)_

### Demo

[![Watch the demo](images/vias_demo_thumbnail.png)](https://www.youtube.com/watch?v=m4j9gsRIXTE)

---

## Getting Started

### Overview Usage
The framework consists of the following tools:

1. **Map Extraction Tool (`met.py`)** (Optional)
   - Extract semantic, street, and building height data from OpenStreetMap (OSM).
   - Extract radio signal data from OpenCellID (OCID).

2. **Map Creation Tool (`mct.py`)** (Optional)
   - Create 3D grid maps (e.g., risk maps, noise maps, obstacle maps) using OSM data.
   - Generate a 3D radio disturbance map using OCID data.

3. **Path Planning Tool (`mopp.py`)**
   - Compute a Pareto set of trade-off paths connecting the given start and goal points.

4. **Path Visualization Tool (`pvt.py`)** (Optional)
   - Visualize the resulting paths.
   - Convert between path formats.

For detailed information on the usage of the tools, see the section [Details Usage](#details-usage).

### Installation

**Setup folder structure**

1. Create an empty working folder, in the following called `workspace`
2. Pull this repo into the `workspace`
3. Create another folder `VIAS_data` and create the following structure (alternatively you can pull https://github.com/NikHoh/VIAS_data that already contains exemplary data)

The project's folder structure will look like this:
    
    ```
    |-- workspace
      |-- VIAS_data
      |   |-- input
      |   |   |-- osm
      |   |   |-- ocid
      |   |-- processing
      |   |-- output
      |   |-- scenario_info.json
      |-- VIAS (this repo)
      |   |-- src
      |   |   |-- vias
      |   |   |   |-- ...
      |   |-- tests
      |   |   |-- ... 
      |   |-- examples
      |   |   |-- ... 
      |   |-- README.md (this readme)
      |   |-- pyproject.toml   
      |   |-- ...
      
    ```

Now you have two options to proceed: 

**Option A: Build by yourself**

1. Install Miniconda (Python environment management tool) following https://docs.anaconda.com/miniconda/
2. To use the Map Extraction Tool (MET), install Osmosis following https://wiki.openstreetmap.org/wiki/Osmosis/Installation
3. Install a Python virtual environment with `conda`
   - `conda env create --name vias_env`
4. Activate the environment
   - `conda activate vias_env`
5. Navigate into the `VIAS` folder
   - `cd workspace/VIAS` 
6. Install VIAS
    - `pip install -e .`

**Option B: Build and use Docker container**

1. Install Docker Engine following https://docs.docker.com/engine/install/
2. Navigate into the `VIAS` folder
   - `cd workspace/VIAS`
3. Build Docker container
   - `docker build -t vias:latest .`


### Details Usage

The four tools (Map Extraction Tool, Map Creation Tool, Path Planning Tool, Path Visualization Tool) are explained in more detail, following an exemplary path planning task set in the city of Paris.

**All data that is used in the following example, can also be obtained from https://github.com/NikHoh/VIAS_data**.

All four tools take scenario information as input that is provided within the `scenario_info.json` tile in the `VIAS_data` 
folder. The scenario information given in the [VIAS_data repo](https://github.com/NikHoh/VIAS_data) can be adapted to arbitrary parameters (e.g., operation spaces).

#### Scenario Information

The `scenario_info.json` file contains information about
  - the coordinates (longitude `map_NW_origin_lon` & latitude `map_NW_origin_lat`) of the operation space north-west corner
  - the coordinates (longitude `tmerc_proj_origin_lon` & latitude `tmerc_proj_origin_lat`) of the [Transverse Mercator](https://en.wikipedia.org/wiki/Transverse_Mercator_projection) projection center
    - the projection is used to transform Decimal degree (WGS84) (i.e., longitude & latitude coordinates) into a Cartesian (East, North) frame
    - if not sure about this, just set it to the same values as `map_NW_origin_lon` & `map_NW_origin_lat`
  - the dimensions `x_length`, `y_length`, and `z_length` of the operation space in x-, y-, and z-direction in meter
  - the resolutions `x_res`, `y_res`, and `z_res` of the operation space in x-, y-, and z-direction in meter
  - a `user_identifier` string for the scenario
  - the minimum allowed flight height `min_flight_height`
  - the maximum allowed flight height `max_flight_height`

##### Restrictions

There are some restrictions to the parameters:
  - `x_length`, `y_length`, and `z_length` must be properly dividable by `x_res`, `y_res`, and `z_res` respectively
  - `min_flight_height` and `max_flight_height` must be properly dividable by `z_res`
  - if the scenario dimensions are too big or the resolution too granular, the respective grid maps will need large storage
    - if the expected storage for one grid map exceeds 750MB, execution is terminated


#### Map Extraction Tool (`met.py`):

The Map Extraction Tool (MET) takes as input:
  - `ScenarioInfo`
  - raw data from OpenStreetMap (OSM) 
  - optionally raw data from OpenCELLiD (OCID) 

It creates the following output:
  
From the OSM data, it creates:
  - filtered versions of the original `*osm.bz2` data in the folder `osm` that only contain the OSM data for the defined operation space on lon-lat coordinates containing
    - all OSM information
    - the filtered semantic information
    - the filtered street information
    - the filtered building height information

  - a `*.pkl` file in the folder `city_models` containing a structure called `CityModel` containing
    - the given scenario information `ScenarioInfo`
    - geospatial positions is [tmerc coordinates](https://en.wikipedia.org/wiki/Transverse_Mercator_projection) of
      - streets 
      - buildings
      - other semantic elements (e.g., parks)
  - `*.pkl` files in the folder `grid_maps` being two-dimensional grid map (i.e., matrices or two-dimensional discrete scalar fields) representations of the extracted information on
    - streets
    - building heights
    - other semantic elements (e.g., parks)
    - Optional: radio signal positions in case OCID data is provided (as described in the following)  

Optionally from the OCID data, it creates:
  - a `*.csv` file in the folder `ocid` containing the [tmerc coordinates](https://en.wikipedia.org/wiki/Transverse_Mercator_projection) of extracted radio cell positions 
  - a `*.pkl` file in the folder `radio_towers` containing the local x-y-coordinates of the extracted radio cell positions

To run the MET example:

1. Download OSM data in the `*.osm.bz2` format or the `*.osm.pbf` format for the desired operation space
   - e.g. from https://download.geofabrik.de/ for larger areas (Osmosis is faster for smaller areas) or from https://extract.bbbike.org/ for specific areas
   - Info: The MET works with the `*.osm.bz2` format. To convert from `*.osm.pbf` use Osmosis:
      - In case you followed installation A (Build yourself):
         - ```osmosis --read-pbf myfile.osm.pbf --write-xml myfile.osm.bz2```
      - In case you followed installation B (Docker container):
         - Enter the Docker container's bash `docker run -it -v <path_to_workspace>/VIAS_data:/VIAS_data vias:latest /bin/bash`
         - Navigate to the folder where the *.osm.pbf file is located
         - ```osmosis --read-pbf myfile.osm.pbf --write-xml myfile.osm.bz2```
   - More info on the conversion: https://download.geofabrik.de/bz2.html
   - place the `*.osm.bz2` file in the folder `workspace/VIAS_data/input/osm`

2. Download OCID data in the `*.csv.gz` format for the desired operation space from https://opencellid.org/
   - place the file in the folder `workspace/VIAS_data/input/ocid`

3. Depending on the installation option:
   - In case you followed installation A (Build yourself):
      - Navigate into the example script folder
         - `cd workspace/VIAS/vias/examples`
      - Enter the virtual environment if not activated yet
         - `conda activate vias_env`
      - Run the map extraction tool (MET)
         - `python met_example.py`
   - In case you followed installation B (Docker container):
      - Mount data folder and run Docker container
         - `docker run -it  -v <path_to_workspace>/VIAS_data:/VIAS_data vias:latest python3 met_example.py`
4. The tool should produce the different mentioned outputs in the respective folders along with images in the folder `workspace/VIAS_data/input/grid_map_plots` visualizing
   - semantic data of the operation space
   - road traffic streets underneath the operation space
   - heights of buildings underneath the operation space
   - the positions of radio signals

To run the MET with your own scenario data:

Either
   - adapt the data in the `met_example.py` file accordingly and run it
or 
   - directly run `met.py` and providing as arguments the paths to 
     - the osm file, 
     - the base data folder,
     - the config file
     - the data save folder, and
     - and optionally to the ocid files
   - ensuring that the base data folder contains a `scenario_info.json` file.
   - Hint: Using the Docker container 
     - you can enter the container's bash
        - `docker run -it -v <path_to_workspace>/VIAS_data:/VIAS_data vias:latest /bin/bash`
     - Navigate to the `vias` folder
     - and run 
       - `python3 met.py <args>`

#### Map Creation Tool (`mct.py`):

The Map Creation Tool (MCT) takes as input:
  - the `ScenarioInfo`
  - the different grid maps that were produced by the MET:
    - buildings (height) grid map
    - streets grid map
    - semantic grid map
    - radio signal tower grid map
  - a set of parameters defined in the file `mct_config.yaml` defining
    - which maps to create, and
    - a set of parameters for each respective map creator

According to the different map creators that are defined in the folder `VIAS/src/vias/map_creators`, the MCT creates different grid maps:

  - clearance height grid map
    - 2D matrix indicating a save flight altitude over buildings
  - minimum flight height grid map
    - 3D binary matrix indicating grid cells below the minimum flight height with `1`
  - obstacle grid map
    - 3D matrix indicating grid cells within static obstacles (+ safety distance) with `1`
  - NFA (no-fly-area) grid map
    - combining obstacle and minimum flight height grid maps
  - risk grid map
    - 3D matrix with risk values for each grid cell, indicating the potential risk of injury to city residents if a UAV flying through that cell experiences a malfunction
    - following the theoretical preliminaries given in the section [Scientific Foundation](#scientific-foundation)
  - radio disturbance grid map
    - 3D matrix with signal disturbance values for each grid cell depending on the radio cell tower positions
    - following the theoretical preliminaries given in the section [Scientific Foundation](#scientific-foundation)
  - noise grid map
    - 3D matrix with noise values for each cell depending on the altitude of the cell and its distance to ground traffic streets
    - following the theoretical preliminaries given in the section [Scientific Foundation](#scientific-foundation)

Own grid maps can be defined using custom map creator classes inheriting from the `MapCreator` class and 
adding them to the config file `mct_config.yaml`.

To run the MCT example:

1. Run the map extraction tool (MET), if not done yet
   - see section [Map Extraction Tool (`met.py`)](#map-extraction-tool-metpy)
2. Depending on the installation option:
   - In case you followed installation A (Build yourself):
     - Navigate into the example script folder
        - `cd workspace/VIAS/vias/examples`
     - Enter the virtual environment if not activated yet
        - `conda activate vias_env`
     - Run the map creation tool (MCT)
       - `python mct_example.py`
   - In case you followed installation B (Docker container):
     - Mount data folder and run Docker container
        - `docker run -it  -v <path_to_workspace>/VIAS_data:/VIAS_data vias:latest python3 mct_example.py`
3. The tool should produce the different grid maps in the folder `grid maps` along with images in the folder `workspace/VIAS_data/input/grid_map_plots` visualizing the generated grid maps in different ways (e.g. flat layer plot, plot of all slices and volume plot)

Info: For large grid maps, the plotting of volume and slices may take too much time. You can turn off the plotting by setting `suppress_grid_image_save` and `suppress_grid_image_plot` to True in the `mct_config.yaml` file.

To run the MCT with your own scenario data:

Either
   - adapt the data in the `mct_example.py` file accordingly and run it
or 
   - directly run `mct.py` and providing as arguments the paths to
     - the base data folder,
     - the config file,
     - the data input folder, and
     - the data save folder
   - ensuring that the base data folder contains a `scenario_info.json` file.
   - Hint: Using the Docker container 
      - you can enter the container's bash
         - `docker run -it -v <path_to_workspace>/VIAS_data:/VIAS_data vias:latest /bin/bash`
      - Navigate to the `vias` folder
      - and run 
         - `python3 mct.py <args>`

#### Multi-objective Path Planning (`mopp.py`):

The Multi-objective Path Planning (MOPP) takes as input:
  - the `ScenarioInfo`
  - the different grid maps that were produced by the MCT:
    - clearance height grid map
    - nfa grid map
    - risk grid map
    - radio disturbance grid map
    - noise grid map
  - a start point for the path to be planned in lon-lat coordinates
  - a goal point for the path to be planned in lon-lat coordinates
  - a set of parameters defined in the file `mopp_config.yaml` defining
    - which objective functions and constraints to use for the optimization
    - a set of parameters for each respective simulator and constraint checker

According to the different simulators (objective functions) defined in the folder `VIAS/src/vias/simulators` 
and constraint checkers (constraint functions) defined in the folder `VIAS/src/vias/constraint_checkers`, the MOPP 
framework optimizes paths regarding different objective functions and subject to different constraint functions.

The so-called grid-based simulators `simulator_grid_based.py` evaluate the quality/cost of a path by calculating its line integral over the
respective grid map along the path (the line integral is approximated as trapezoidal sum). In the given example, this
is true for:
  - the risk grid map
  - the noise grid map
  - the radio disturbance grid map

Any other simulator/objective function that can not be described by a line integral, e.g. that does not work on a grid 
map belongs to the class of non-grid-based/non-graph-based objective functions. In the given example, this is true for:
  - the energy simulator `simulator_energy.py` 

Among the defined constraint functions, the
  - `constraint_checker_minimum_flight_height.py` punishes path waypoints that lie below the minimum flight height,
  - `constraint_checker_out_of_operation_space.py` punishes path waypoints that lie out of the operation space, and
  - `constraint_checker_static_obstacle_collision.py` punishes path waypoints that lie within static obstacles.

For more information on the theoretical background of the objective functions and constraints, see the literature referenced
in section [Scientific Foundation](#scientific-foundation).

Own objective functions/simulators can be defined using custom simulator classes inheriting from the `Simulator` class defined in `simulator.py`
and adding them to the config file `mopp_config.yaml`.

Own constraint functions can be defined using custom constraint checker classes inheriting from the `ConstraintChecker` class defined in `constraint_checker.py`
and adding them to the config file `mopp_config.yaml`.

To run the MOPP example:

1. Run the map extraction tool (MET), if not done yet
   - see section [Map Extraction Tool (`met.py`)](#map-extraction-tool-metpy)
2. Run the map creation tool (MCT), if not done yet
   - see section [Map Creation Tool (`mct.py`)](#map-creation-tool-mctpy)
3. Depending on the installation option:
   - In case you followed installation A (Build yourself):
     - Navigate into the example script folder
        - `cd workspace/VIAS/vias/examples`
     - Enter the virtual environment if not activated yet
        - `conda activate vias_env`
     - Run the multi-objective path planning (MOPP)
        - `python mopp_example.py`
   - In case you followed installation B (Docker container):
     - Mount data folder and run Docker container
        - `docker run -it -v <path_to_workspace>/VIAS_data:/VIAS_data vias:latest python3 mopp_example.py`
4. The MOPP should produce
   - graph representations of the used grid maps in the folder `workspace/VIAS_data/processing/grid_graphs`
   - results of the pre-processing (Dijkstra) module in the folder `workspace/VIAS_data/processing/dijkstra_paths`. This includes visualizations of the
     - raw Dijkstra paths
     - the smoothed Dijkstra paths
     - and the approximated (as NURBS curve) paths, i.e. the initial paths that go into the meta-heuristic (evolutionary) optimization step
5. After successfully terminating, MOPP has created several output files in the respective auto-generated output folder `workspace/VIAS_data/output/<xyz>`: 
   - a `scenario_info.json` file with the saved scenario info
   - different formats (`*.html` `plotly` visualization, `*.png` plot, and raw statistics data as `*.pkl`) of the optimization (convergence) statistics
   - a `mopp_statistics.json` file containing (mostly) run-time-related information
   - a 3D path visualization of the optimized Pareto set of paths (`*.html` `plotly` visualization, `*.png` plot)
   - a `F_X_dict.pkl` file containing the ojbective function values (`F`) and the optimization variables (`X`) (i.e., the control point positions) of all Pareto set paths
   - a folder `optimized_paths` that contains visualizations of the Pareto set's extreme point solutions (i.e., the 'optimal' solutions regarding one objective function)

To run the MOPP tool with your own scenario data:

Either
   - adapt the data in the `mopp_example.py` file accordingly and run it
or 
   - directly run `mopp.py` and providing as arguments
     - the path to the base data folder,
     - the longitude coordinate to the path's start,
     - the latitude coordinate to the path's start,
     - the longitude coordinate to the path's goal,
     - the latitude coordinate to the path's goal,
     - the path to the config file,
     - the path to the data input folder,
     - the path to the data save folder, and
     - the path to the data processing folder
   - ensuring that the base data folder contains a `scenario_info.json` file.
   - Hint: Using the Docker container 
      - you can enter the container's bash
         - `docker run -it -v <path_to_workspace>/VIAS_data:/VIAS_data vias:latest /bin/bash`
      - Navigate to the `vias` folder
      - and run 
         - `python3 mopp.py <args>`

#### Path Visualization Tool (`pvt.py`):

The Path Visualization Tool (PVT) takes as input:
  - a set of parameters defined in the file `pvt_config.yaml` defining
    - map dimensions to be able to load the correct maps by the defined naming convention
  - the path to the input grid maps
  - the path to the optimization output folder that should at least contain
    - the scenario info `scenario_info.json`
    - the MOPP statistics `mopp_statistics.json`
    - the Pareto set data `F_X_dict.pkl`
    - the optimization statistics data `optimizer_statistics.pkl`

The functions `process_optimizer_statistics()` and `process_F_X_dict_()` can be adapted to serve arbitrary visualization purposes.
Currently, `process_optimizer_statistics()` loads the `optimizer_statistics.pkl` file and plots them as `*.html` plotly plot and `*.png` (as already done at the end of a successful mopp.py run).
Then, it prints the last racked ideal point of the Pareto set.
The function `process_F_X_dict_()` loads the `F_X_dict.pkl` file and prints the knee point's objective function values.
Then, it plots the three-dimensional knee point path and stores the plot in the respective auto-generated output folder `workspace/VIAS_data/output/<xyz>`. 

To run the PVT example:

1. Run the map extraction tool (MET), if not done yet
   - see section [Map Extraction Tool (`met.py`)](#map-extraction-tool-metpy)
2. Run the map creation tool (MCT), if not done yet
   - see section [Map Creation Tool (`mct.py`)](#map-creation-tool-mctpy)
3. Run the multi-objective path planning (MOPP), if not done yet
   - see section [Multi-objective Path Planning (`mopp.py`)](#multi-objective-path-planning-mopppy)
4. Depending on the installation option:
   - In case you followed installation A (Build yourself):
     - Navigate into the example script folder
        - `cd workspace/VIAS/vias/examples`
     - Enter the virtual environment if not activated yet
        - `conda activate vias_env`
     - Run the path visualization tool (PVT)
        - `python pvt_example.py`
   - In case you followed installation B (Docker container):
     - Mount data folder and run Docker container
        - `docker run -it -v <path_to_workspace>/VIAS_data:/VIAS_data vias:latest python3 pvt_example.py`
5. The PVT should produce
   - some console output information in the Pareto set's ideal and knee point
   - a 3D path plot of the knee point solution

To run the PVT with your own scenario data:

Either
   - adapt the data in the `pvt_example.py` file accordingly and run it
or 
   - directly run `pvt.py` and providing as arguments the paths to
     - the base data folder,
     - the config file,
     - the optimization results, and
     - and optionally the path to the grid maps' folder
   - ensuring that the base data folder contains a `scenario_info.json` file.
   - Hint: Using the Docker container 
      - you can enter the container's bash
         - `docker run -it -v <path_to_workspace>/VIAS_data:/VIAS_data vias:latest /bin/bash`
      - Navigate to the `vias` folder
      - and run 
         - `python3 pvt.py <args>`

### Crucial Parameters

TBA


### Common Issues and Solutions

1. The **Niching** feature ensures that pre-processed Dijkstra solutions being initial solutions in the meta-heuristic (evolutionary) optimization are not rejected due to constraint violations induced by the representation shift from discrete to continuous.
  Therefore, every initial solution is treated in a separate niche (the evolutionary variation and selection scheme happens separately in a niche independent of the other niches) until at least one individual per niche satisfies all constraints. If that is the case
  all individuals are resolved into one single population and the optimization process continues.

  In case the optimization terminates while Niching is still active, an `OptimizationEndsInNiches` exception is thrown and no valid output data is generated.

  To avoid this:
   - Make sure no unsatisfiable constraints exist. This may, for example, be the case, if the start and end point of the path are set too close to the operation space boundaries. 
   - Increase the iterations to increase the changes of finding at least one solution (i.e., path) per niche that satisfies all constraints.

 

---

## Upcoming Features

Planned updates in the next two to three weeks:
- Documentation
- Enhanced visualization features (e.g., export to Google Earth Studio)
- Docker environment setup

---
	
## References

- [IEEE Journal Paper](https://ieeexplore.ieee.org/document/10196046)
- [IEEE Conference Paper](https://ieeexplore.ieee.org/document/9870265)
- [Dissertation](https://tuprints.ulb.tu-darmstadt.de/28839/)

---

## Changelog
For a detailed history of changes, see the [Changelog](CHANGELOG.md).

---

## License

This project is licensed under the **BSD 3-Clause License**. See the [LICENSE](LICENSE) file for details.