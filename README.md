<img src="images/vias_logo.svg">

# VIAS

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

3. **PhD Thesis:** Details to be announced (TBA).

### Demo

[![Watch the demo](images/vias_demo_thumbnail.png)](https://www.youtube.com/watch?v=m4j9gsRIXTE)

---

## Getting Started

### Installation
Installation instructions will be provided soon together with a Docker container.

### Usage
The framework consists of the following tools:

1. **Map Extraction Tool (`met.py`)** (Optional)
   - Extract semantic, street, and building height data from OpenStreetMap (OSM).
   - Extract radio cell tower data from OpenCellID (OCID).

2. **Map Creation Tool (`mct.py`)** (Optional)
   - Create 3D grid maps (e.g., risk maps, noise maps, obstacle maps) using OSM data.
   - Generate a 3D radio disturbance map using OCID data.

3. **Path Planning Tool (`mopp.py`)**
   - Compute a Pareto set of trade-off paths connecting the given start and goal points.

4. **Path Visualization Tool (`pvt.py`)** (Optional)
   - Visualize the resulting paths.
   - Convert between path formats.

---

## Upcoming Features

Planned updates in the next two to three weeks:
- Detailed README with:
  - System requirements
  - Installation guide
- Full documentation
- Integration with missing third-party libraries
- Enhanced visualization features (e.g., export to Google Earth Studio)
- Docker environment setup
- Video visualization of results

---
	
## References

- [IEEE Journal Paper](https://ieeexplore.ieee.org/document/10196046)
- [IEEE Conference Paper](https://ieeexplore.ieee.org/document/9870265)

---

## Changelog
For a detailed history of changes, see the [Changelog](CHANGELOG.md).

---

## License

This project is licensed under the **BSD 3-Clause License**. See the [LICENSE](LICENSE) file for details.