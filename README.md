<div align="center"> <h1> Implementing IsoGeometric Analysis with NURBS in fdaPDE </h1>

<h5> fdaPDE - Physics-Informed Spatial and Functional Data Analysis </h5> </div>

This repository is a fork of the fdaPDE-core C++ header only library system for the fdaPDE project. The main purpose of this fork is the developement of the final project for the course of Advanced Programming for Scientific Computing (APSC), held by professor Luca Formaggia at Politecnico di Milano. This project was developed by two students, Angelo Curti (103337) and Lorenzo De Gaspari (216519), under the supervision of professor Laura M. Sangalli and the tutorship of professor Eleonora Arnone and doctor Alessandro Palummo.
Our main contribution to the repository is the implementation of the IsoGeometric Analysis discretization method from scratch. Most of the code is located in the fdaPDE/isogeometric_analysis directory.

## Documentation
The official documentation of fdaPDE can be found on [documentation site](https://fdapde.github.io/)

## Dependencies
fdaPDE-core is an header-only library, therefore it does not require any installation. Just make sure to have it in your include path. Neverthless, compiled code including the core library must satisfy the following dependencies:

- **C++17 compliant compiler**  
  We are using gcc 11.2.0, but any version higher than 7 should be enough.

- **make**

- **CMake**

- **Eigen3** linear algebra library  
  Version 3.3 or newer (we are using 3.3.9).

- **gnuplot** (optional)  
  Only required to create time and convergence plots.

- **gtest** (optional)  
  Only needed to run the library tests.

## How to run the code

- **Library tests**
    The library was already equipped with tests to ensure functional correctness, and we added additional tests to confirm that our code works as expected and hasn’t impacted any existing features. They can be executed using the following commands from the directory in which you've cloned the library.
    
    ```bash
    #!/bin/bash
    cd fdaPDE-core/test
    ./run_tests.sh
    ```

- **Simulations**
    All simulations presented in Chapter 4 of our report can be run directly from shell, taking as example the square laplacian simulation, with the following commands:
    
    ```bash
    #!/bin/bash
    cd fdaPDE-core/simulations
    cd laplacian-square
    cmake CMakeLists.txt
    make
    ./run_test
    ```

    To run other simulations, simply substitute line 3 with the directory associated to the desired simulation. The possible options are:
    * diffusion_advection-ring,
    * diffusion_reaction-square,
    * laplacian-ring,
    * laplacian-square,
    * laplacian_curlyplate,
    * neumann_curlyplate.

    To visualize plots, execute the following command in the simulation folder. It will create a directory called "outputs/" and save the plots inside of it as .png files.
    
    ```bash
    #!/bin/bash
    ./make_plots.sh
    ```

    Be mindful when running the diffusion_advection-ring test, as it may take around 20 minutes to complete.