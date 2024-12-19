# Changelog

## [0.1.0] - 2024-11-22
### Initial Release
- Initial commit of the VIAS project, a many-objective path planning framework for unmanned aerial vehicles (UAVs) in urban environments.
- Basic functionality includes:
  - Path planning for UAVs over city environments based on multiple objective functions (risk, noise, energy, etc.).
  - Use of georeferenced data from OpenStreetMap (OSM) and OpenCellID (OCID) for path optimization.
  - Includes tools for map extraction and creation (from OSM and OCID).
  - Ability to calculate and visualize Pareto sets of trade-off solutions for UAV flight paths.
  - Example objective functions provided (e.g., minimize energy, noise, risk, and signal disturbance).
- Initial commit of the project, with more features planned in future releases.

## [0.1.1] - 2024-12-18
### Functional Release
- 3rd party library integration
- Installation and usage guide in README.md
- pytest integration
- CI/CD pipeline integration

---

## Upcoming Changes
- Future releases will include:
  - Documentation
  - Additional features for advanced visualization
  - Docker environment setup.
